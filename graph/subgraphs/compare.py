"""
graph/subgraphs/compare.py

cmp_subgraph: 두 개 이상 제품 비교 처리

검색 전략 (방식 C — aspect 컬럼 직접 필터):
  ① axes 추출 (LLM, CompareAxisResult structured_output)
  ② axes → aspect 컬럼 매핑
  ③ 제품 × axis 조합으로 aspect 필터 + 벡터 유사도 검색
     - axis당 positive 리뷰 k=3개 + negative 리뷰 k=2개
     - aspect 없는 axis(향 등)는 일반 벡터 검색 fallback
     - skin_type 필터 포함, 결과 없으면 필터 제거 fallback
  ④ 제품별 dedup → 상위 15개
  ⑤ 제품별 CompareSummaryResult structured_output
  ⑥ 축별 winner/draw 판정
     - DB 컬럼이 있는 기준(밀림·보습 등) → product_stats 긍정률로 코드가 판정 (LLM 없음)
     - DB 컬럼이 없는 기준(향 등)만 → StructuredCompareResult로 LLM이 판정

  2026-11 수정:
    이전에는 모든 기준을 LLM이 요약문만 보고 판정했다. 요약에 쓰인 리뷰는
    기준마다 긍정 2개 + 부정 2개로 일부러 같은 수를 뽑은 것이라, 실제 긍정률이
    95%인 제품도 LLM 눈에는 '반반'으로 보였다.
    (예: 비플레인 녹두 보습 95% vs 셀퓨전씨 72%인데 셀퓨전씨 승으로 판정)
    → DB에 집계된 기준은 긍정률 숫자로 판정하고, 근거 문장에도 그 숫자를 넣는다.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document

from graph.state import GraphState
from graph.prompts import (
    CompareAxisResult,
    CompareSummaryResult,
    StructuredCompareResult,
    AxisAnalysis,
    AxisJudgmentItem,
    COMPARE_AXIS_SYSTEM, COMPARE_AXIS_HUMAN,
    COMPARE_SUMMARY_SYSTEM, COMPARE_SUMMARY_HUMAN,
    STRUCTURED_COMPARE_SYSTEM, STRUCTURED_COMPARE_HUMAN,
)
from rag.retriever import (
    SKIN_TYPE_MAP,
    AXES_TO_ASPECT,
    load_product_stats,
    search_by_aspect,
    search_by_goods_no,
    resolve_goods_no,
)

logger = logging.getLogger(__name__)

from graph.llm import get_llm

MAX_DOCS_PER_PRODUCT = 15
POS_PER_AXIS = 2  # 속도 최적화: 3 → 2
NEG_PER_AXIS = 2  # 정확도 유지 (부정 리뷰는 2개 유지)

# DB 판정 기준
MIN_MENTION = 5      # 화면 표시 기준과 동일: 언급 5건 미만이면 판정에 쓰지 않음
DRAW_GAP    = 0.05   # 1·2위 긍정률 차이가 5%p 미만이면 무승부


def run_compare_subgraph(state: GraphState) -> dict:
    llm            = get_llm()
    query          = _get_rewritten_question(state)
    products       = state.get("products_mentioned", [])
    skin_type_str  = state.get("skin_type", "")
    skin_type_code = SKIN_TYPE_MAP.get(skin_type_str, "")

    # ── ① axes 추출 ──────────────────────────────────────────────────────────
    try:
        axis_result: CompareAxisResult = llm.with_structured_output(
            CompareAxisResult
        ).invoke([
            {"role": "system", "content": COMPARE_AXIS_SYSTEM},
            {"role": "human",  "content": COMPARE_AXIS_HUMAN.format(
                question=query,
                products=", ".join(products),
            )},
        ])
        axes = axis_result.axes
    except Exception as e:
        logger.warning("axes 추출 실패: %s", e)
        axes = ["보습력", "자극", "발림성"]

    # ── ② axes → aspect 컬럼 매핑 ────────────────────────────────────────────
    axis_aspect_map: dict[str, str | None] = {
        ax: AXES_TO_ASPECT.get(ax) for ax in axes
    }

    # ── ③ 제품 × axis 검색 ───────────────────────────────────────────────────
    # all_docs_seen: 전체 dedup (retrieved_docs 저장용)
    # product_seen:  제품별 dedup (요약 품질 보장용) — 제품 루프 안에서 초기화
    all_docs: list[Document] = []
    all_docs_seen: set[str] = set()
    product_doc_map: dict[str, list[Document]] = {}

    for product_name in products:
        goods_no = resolve_goods_no(product_name)
        if not goods_no:
            logger.warning("goods_no 조회 실패: %s — 해당 제품 비교에서 제외", product_name)
            continue

        product_bucket: list[Document] = []
        product_seen:   set[str]       = set()   # 제품별 독립 dedup

        for ax, aspect in axis_aspect_map.items():
            if aspect is not None:
                # positive 리뷰
                try:
                    for doc in search_by_aspect(
                        query=ax, goods_no=goods_no, aspect=aspect,
                        skin_type_code=skin_type_code,
                        sentiment="positive", k=POS_PER_AXIS,
                    ):
                        doc_id = doc.metadata.get("id", doc.page_content[:50])
                        if doc_id not in product_seen:
                            product_seen.add(doc_id)
                            product_bucket.append(doc)
                        if doc_id not in all_docs_seen:
                            all_docs_seen.add(doc_id)
                            all_docs.append(doc)
                except Exception as e:
                    logger.warning("search_by_aspect pos(%s, %s): %s", product_name, ax, e)

                # negative 리뷰 (실패 포인트 파악)
                try:
                    for doc in search_by_aspect(
                        query=ax, goods_no=goods_no, aspect=aspect,
                        skin_type_code=skin_type_code,
                        sentiment="negative", k=NEG_PER_AXIS,
                    ):
                        doc_id = doc.metadata.get("id", doc.page_content[:50])
                        if doc_id not in product_seen:
                            product_seen.add(doc_id)
                            product_bucket.append(doc)
                        if doc_id not in all_docs_seen:
                            all_docs_seen.add(doc_id)
                            all_docs.append(doc)
                except Exception as e:
                    logger.warning("search_by_aspect neg(%s, %s): %s", product_name, ax, e)
            else:
                # aspect 매핑 없는 axis(향 등) → 일반 벡터 검색 fallback
                try:
                    for doc in search_by_goods_no(
                        query=ax, goods_no=goods_no,
                        skin_type_code=skin_type_code, k=POS_PER_AXIS,
                    ):
                        doc_id = doc.metadata.get("id", doc.page_content[:50])
                        if doc_id not in product_seen:
                            product_seen.add(doc_id)
                            product_bucket.append(doc)
                        if doc_id not in all_docs_seen:
                            all_docs_seen.add(doc_id)
                            all_docs.append(doc)
                except Exception as e:
                    logger.warning("fallback search(%s, %s): %s", product_name, ax, e)

        product_doc_map[product_name] = product_bucket[:MAX_DOCS_PER_PRODUCT]

    # ── ④ 제품별 요약 생성 ───────────────────────────────────────────────────
    product_summaries: list[dict] = []

    def _summarize_one(product_name: str, pdocs: list) -> dict:
        """단일 제품 요약 — ThreadPoolExecutor로 병렬 실행"""
        if not pdocs:
            return {
                "product_name": product_name,
                "goods_no": resolve_goods_no(product_name) or "",
                "axes": {ax: "리뷰 데이터 부족" for ax in axes},
                "best_for": "",
                "avoid_if": "",
            }
        docs_text = _format_docs(pdocs)
        try:
            summary: CompareSummaryResult = llm.with_structured_output(
                CompareSummaryResult
            ).invoke([
                {"role": "system", "content": COMPARE_SUMMARY_SYSTEM},
                {"role": "human",  "content": COMPARE_SUMMARY_HUMAN.format(
                    product_name=product_name,
                    skin_type=skin_type_str or "미입력",
                    axes=", ".join(axes),
                    docs=docs_text,
                )},
            ])
            d = summary.model_dump()
            d["goods_no"] = resolve_goods_no(product_name) or ""
            return d
        except Exception as e:
            logger.warning("제품 요약 생성 실패(%s): %s", product_name, e)
            return {
                "product_name": product_name,
                "goods_no": resolve_goods_no(product_name) or "",
                "axes": {ax: "요약 생성 실패" for ax in axes},
                "best_for": "",
                "avoid_if": "",
            }

    # 제품별 요약을 병렬로 실행 (달바 + 라운드랩 동시에)
    items = list(product_doc_map.items())
    with ThreadPoolExecutor(max_workers=min(len(items), 4)) as executor:
        futures = {
            executor.submit(_summarize_one, pname, pdocs): pname
            for pname, pdocs in items
        }
        results_map = {}
        for future in as_completed(futures):
            pname = futures[future]
            results_map[pname] = future.result()

    # 원래 순서 유지
    product_summaries = [results_map[pname] for pname, _ in items if pname in results_map]

    # ── ⑤ 축별 winner 판정 ───────────────────────────────────────────────────
    structured_compare_result: dict = {}
    if len(product_summaries) >= 2:
        # DB 컬럼이 있는 기준 → 코드가 긍정률로 판정
        db_judgments = {
            ax: judge_axis_by_stats(ax, aspect, product_summaries)
            for ax, aspect in axis_aspect_map.items()
            if aspect is not None
        }

        # DB 컬럼이 없는 기준(향 등)만 LLM 판정
        llm_axes = [ax for ax, aspect in axis_aspect_map.items() if aspect is None]
        llm_judgments: dict[str, dict] = {}
        if llm_axes:
            summaries_text = _format_summaries_for_compare(product_summaries)
            try:
                compare_result: StructuredCompareResult = llm.with_structured_output(
                    StructuredCompareResult
                ).invoke([
                    {"role": "system", "content": STRUCTURED_COMPARE_SYSTEM},
                    {"role": "human",  "content": STRUCTURED_COMPARE_HUMAN.format(
                        question=f"{query}\n(판정할 기준: {', '.join(llm_axes)})",
                        summaries=summaries_text,
                    )},
                ])
                for j in compare_result.model_dump()["judgments"]:
                    if j.get("axis") in llm_axes:
                        llm_judgments[j["axis"]] = {**j, "basis": "리뷰 요약(LLM)"}
            except Exception as e:
                logger.warning("structured_compare 실패: %s", e)

        # 질문에서 뽑은 기준 순서 유지 (물은 기준이 첫 행)
        judgments = []
        for ax in axes:
            j = db_judgments.get(ax) or llm_judgments.get(ax)
            if j:
                judgments.append(j)
        structured_compare_result = {"judgments": judgments}

    return {
        "retrieved_docs":            all_docs,
        "product_summaries":         product_summaries,
        "compare_axes":              axes,
        "structured_compare_result": structured_compare_result,
    }


# ── DB 기반 판정 ─────────────────────────────────────────────────────────────

def judge_axis_by_stats(axis: str, aspect: str, product_summaries: list[dict]) -> dict:
    """
    product_stats의 '{aspect}_pos'(긍정률)와 '{aspect}_mention'(언급 수)로 승자 판정

    규칙:
      - 언급 MIN_MENTION건 미만인 제품은 판정에서 제외
      - 판정 가능한 제품이 2개 미만 → 무승부(데이터 부족)
      - 1·2위 긍정률 차이 < DRAW_GAP → 무승부
      - 그 외 → 긍정률 1위가 승자
    reason에는 DB 숫자를 그대로 넣는다 (LLM이 숫자를 만들지 않도록)
    values는 비교표 칸을 코드가 직접 그릴 때 쓴다 (nodes._build_compare_table)
    """
    stats = load_product_stats()
    rows = []
    for s in product_summaries:
        name = s.get("product_name", "")
        st   = stats.get(s.get("goods_no", ""), {})
        rows.append((name, st.get(f"{aspect}_pos"), st.get(f"{aspect}_mention", 0) or 0))
    values = {n: {"pos": p, "mention": m} for n, p, m in rows}

    def fmt(name, pos, mention):
        return f"{name} {int(pos * 100)}%({mention}건)"  # 막대 표시(int)와 같은 방식

    usable = [(n, p, m) for n, p, m in rows if p is not None and m >= MIN_MENTION]
    if len(usable) < 2:
        return {
            "axis": axis, "winner": None, "draw": True,
            "reason": f"{axis} 언급 리뷰가 {MIN_MENTION}건 미만인 제품이 있어 판정 보류",
            "basis": "DB 긍정률", "values": values,
        }

    usable.sort(key=lambda x: x[1], reverse=True)
    detail = " vs ".join(fmt(*u) for u in usable)
    (top_name, top_pos, _), (_, second_pos, _) = usable[0], usable[1]

    if top_pos - second_pos < DRAW_GAP:
        return {
            "axis": axis, "winner": None, "draw": True,
            "reason": f"긍정률 차이 {DRAW_GAP:.0%}p 미만 — {detail}",
            "basis": "DB 긍정률", "values": values,
        }
    return {
        "axis": axis, "winner": top_name, "draw": False,
        "reason": f"긍정률 {detail}",
        "basis": "DB 긍정률", "values": values,
    }


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def _get_rewritten_question(state: GraphState) -> str:
    """검색용 질문: rewritten_question 우선, 없으면 마지막 HumanMessage"""
    if state.get("rewritten_question"):
        return state["rewritten_question"]
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content
    return ""


def _format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        aspect_label = m.get("aspect_label", "")
        label_str    = f" [{aspect_label}]" if aspect_label and aspect_label != "none" else ""
        parts.append(
            f"[{i}]{label_str} 평점:{m.get('score','')} "
            f"피부:{m.get('skin_type','미입력')} "
            f"재구매:{'예' if m.get('is_repurchase') else '아니오'}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)


def _format_summaries_for_compare(summaries: list[dict]) -> str:
    parts = []
    for s in summaries:
        name  = s.get("product_name", "")
        axes  = s.get("axes", [])   # list[{axis, analysis}]
        best  = s.get("best_for", "")
        avoid = s.get("avoid_if", "")
        lines = [f"[{name}]"]
        for item in axes:
            if isinstance(item, dict):
                lines.append(f"  {item.get('axis', '')}: {item.get('analysis', '')}")
        if best:
            lines.append(f"  추천 대상: {best}")
        if avoid:
            lines.append(f"  주의 대상: {avoid}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)