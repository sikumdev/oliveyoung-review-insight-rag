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
  ⑥ StructuredCompareResult — 축별 winner/draw 판정
"""

import logging
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
    search_by_aspect,
    search_by_goods_no,
    resolve_goods_no,
)

logger = logging.getLogger(__name__)

from graph.llm import get_llm

MAX_DOCS_PER_PRODUCT = 15
POS_PER_AXIS = 3
NEG_PER_AXIS = 2


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

    for product_name, pdocs in product_doc_map.items():
        if not pdocs:
            product_summaries.append({
                "product_name": product_name,
                "goods_no": resolve_goods_no(product_name) or "",
                "axes": {ax: "리뷰 데이터 부족" for ax in axes},
                "best_for": "",
                "avoid_if": "",
            })
            continue

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
            product_summaries.append(d)
        except Exception as e:
            logger.warning("제품 요약 생성 실패(%s): %s", product_name, e)
            product_summaries.append({
                "product_name": product_name,
                "goods_no": resolve_goods_no(product_name) or "",
                "axes": {ax: "요약 생성 실패" for ax in axes},
                "best_for": "",
                "avoid_if": "",
            })

    # ── ⑤ 축별 winner 판정 ───────────────────────────────────────────────────
    structured_compare_result: dict = {}
    if len(product_summaries) >= 2:
        summaries_text = _format_summaries_for_compare(product_summaries)
        try:
            compare_result: StructuredCompareResult = llm.with_structured_output(
                StructuredCompareResult
            ).invoke([
                {"role": "system", "content": STRUCTURED_COMPARE_SYSTEM},
                {"role": "human",  "content": STRUCTURED_COMPARE_HUMAN.format(
                    question=query,
                    summaries=summaries_text,
                )},
            ])
            structured_compare_result = compare_result.model_dump()
        except Exception as e:
            logger.warning("structured_compare 실패: %s", e)

    return {
        "retrieved_docs":            all_docs,
        "product_summaries":         product_summaries,
        "compare_axes":              axes,
        "structured_compare_result": structured_compare_result,
    }


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def _get_rewritten_question(state: GraphState) -> str:
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