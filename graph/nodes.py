"""
graph/nodes.py
메인 그래프 노드 정의
"""

import json
from collections import defaultdict

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.types import interrupt

from graph.state import GraphState
from graph.prompts import (
    IntakeResult,
    ProductsExtractResult,
    BatchInsightResult,
    GenerateResult,
    INTAKE_SYSTEM, INTAKE_HUMAN_CONTEXT,
    GENERATE_SYSTEM, GENERATE_GUIDE, GENERATE_HUMAN,
    REWRITE_SYSTEM, REWRITE_HUMAN,
    FALLBACK_SYSTEM, FALLBACK_HUMAN,
    REVIEW_INSIGHT_SYSTEM, REVIEW_INSIGHT_HUMAN,
    SKIN_TYPE_EXTRACT_SYSTEM,
    PRODUCTS_EXTRACT_SYSTEM,
)
from graph.subgraphs.recommend import run_recommend_subgraph
from graph.subgraphs.compare import run_compare_subgraph
from graph.subgraphs.filter import run_filter_subgraph
from rag.retriever import SKIN_TYPE_MAP, load_review_summary

# ── LLM 싱글톤 (graph/llm.py에서 관리) ──────────────────────────────────────
from graph.llm import get_llm


# ── 노드 1: intake_node ──────────────────────────────────────────────────────

def intake_node(state: GraphState) -> dict:
    """
    질문 분류 + 피부타입/제품명/user_needs 추출
    rewritten_question을 HumanMessage로 messages에 추가

    메시지 이력을 문자열 변환 없이 객체 그대로 언패킹해서 전달
    → role 구조 보존, 토큰 효율 향상
    """
    llm            = get_llm()
    structured_llm = llm.with_structured_output(IntakeResult)
    recommended    = state.get("recommended_products", [])

    # 컨텍스트 정보는 system 프롬프트 뒤 별도 SystemMessage로 주입
    context_msg = INTAKE_HUMAN_CONTEXT.format(
        recommended_products=", ".join(recommended) if recommended else "없음",
        selected_products=", ".join(state.get("selected_products", [])) or "없음",
        last_query_type=state.get("last_query_type", "") or "없음",
    )

    result: IntakeResult = structured_llm.invoke([
        SystemMessage(content=INTAKE_SYSTEM),
        SystemMessage(content=context_msg),  # 컨텍스트 주입
        *state["messages"][-6:],             # 대화 이력 그대로 언패킹
    ])

    # rewrite 루프 여부 판단:
    # rewrite_query_node를 타고 온 경우 rewrite_count 유지 (리셋 금지)
    # 새 질문(rewrite_count == 0)이면 0 유지
    is_rewrite = state.get("rewrite_count", 0) > 0

    # follow_up / refine은 이전 skin_type, skin_tone 보존
    # 그 외 새 질문(추천/비교/필터해석)은 새로 추출한 값만 사용
    skin_type = result.skin_type
    skin_tone = result.skin_tone

    if result.query_type in ("follow_up", "refine"):
        if not skin_type:
            skin_type = state.get("skin_type", "")
        if not skin_tone:
            skin_tone = state.get("skin_tone", "")

    # 방어 로직: skin_type이나 skin_tone 있으면 needs_skin_type=False
    needs_skin_type = result.needs_skin_type and not skin_type and not skin_tone

    return {
        "query_type":         result.query_type,
        "needs_skin_type":    needs_skin_type,
        "skin_type":          skin_type,
        "skin_tone":          skin_tone,
        "products_mentioned": result.products_mentioned,
        "user_needs":         result.user_needs,
        "last_query_type":    state.get("query_type", ""),
        "rewrite_count":      state.get("rewrite_count", 0) if is_rewrite else 0,
        "review_insights":    {},
        "messages": [HumanMessage(content=result.rewritten_question)],
    }


# ── 노드 2: ask_skin_type_node ───────────────────────────────────────────────

def _ask_prompt(state: GraphState) -> str:
    """
    상황에 맞는 질문 생성
    - 피부타입 없고 피부톤 있음  → 피부타입만 질문
    - 피부타입 있고 피부톤 없음  → 피부톤만 질문
    - 둘 다 없음                 → 둘 다 질문 (피부톤은 몰라도 됨)
    """
    has_skin_type = bool(state.get("skin_type", ""))
    has_skin_tone = bool(state.get("skin_tone", ""))

    if has_skin_type and not has_skin_tone:
        return "퍼스널컬러나 피부톤을 알고 계신가요? (쿨톤 / 웜톤 / 여름쿨 / 봄웜 / 가을웜 / 모름)"
    if not has_skin_type and has_skin_tone:
        return "피부타입을 알려주세요! (지성 / 건성 / 복합성 / 민감성 / 중성)"
    # 둘 다 없음
    return (
        "피부타입과 피부톤을 알려주시면 더 정확하게 추천해드릴 수 있어요!\n"
        "피부타입: 지성 / 건성 / 복합성 / 민감성 / 중성\n"
        "피부톤: 쿨톤 / 웜톤 / 여름쿨 / 봄웜 / 가을웜 (몰라도 괜찮아요)"
    )


def ask_skin_type_node(state: GraphState) -> dict:
    prompt_msg = _ask_prompt(state)
    user_answer = interrupt(prompt_msg)
    text = str(user_answer)

    # 피부타입 추출
    skin_type = state.get("skin_type", "")
    if not skin_type:
        skin_type = _match_skin_type(text)
        if not skin_type:
            valid_types = list(SKIN_TYPE_MAP.keys())
            response = get_llm().invoke([
                {"role": "system", "content": SKIN_TYPE_EXTRACT_SYSTEM},
                {"role": "human",  "content": text},
            ])
            result = response.content.strip()
            skin_type = result if result in valid_types else ""

    # 피부톤 추출
    skin_tone = state.get("skin_tone", "")
    if not skin_tone:
        skin_tone = _match_skin_tone(text)

    return {
        "skin_type":       skin_type,
        "skin_tone":       skin_tone,
        "needs_skin_type": False,
        "messages":        [HumanMessage(content=text)],
    }


# ── 노드 3: ask_compare_product_node ────────────────────────────────────────

def ask_compare_product_node(state: GraphState) -> dict:
    current = state.get("products_mentioned", [])
    prompt_msg = (
        "비교할 제품을 알려주세요! (예: 달바 vs 라운드랩, 최대 4개)"
        if len(current) == 0
        else f"'{current[0]}'와 비교할 다른 제품을 알려주세요!"
    )
    user_answer = interrupt(prompt_msg)

    # LLM으로 자연어 제품명 추출
    structured_llm = get_llm().with_structured_output(ProductsExtractResult)
    result: ProductsExtractResult = structured_llm.invoke([
        {"role": "system", "content": PRODUCTS_EXTRACT_SYSTEM},
        {"role": "human",  "content": str(user_answer)},
    ])

    # 기존 제품 + 새로 추출된 제품 합치기 (중복 제거, 최대 2개)
    merged = list(dict.fromkeys(current + result.products))[:4]

    return {
        "products_mentioned": merged,
        "messages":           [HumanMessage(content=str(user_answer))],
    }


# ── 노드 4~6: 서브그래프 노드 ────────────────────────────────────────────────

def rec_subgraph_node(state: GraphState) -> dict:
    return run_recommend_subgraph(state)

def cmp_subgraph_node(state: GraphState) -> dict:
    return run_compare_subgraph(state)

def fil_subgraph_node(state: GraphState) -> dict:
    return run_filter_subgraph(state)


# ── 노드 7: review_insight_node ──────────────────────────────────────────────

def review_insight_node(state: GraphState) -> dict:
    """
    retrieved_docs → 제품별 인사이트 추출 (배치 처리)
    with_structured_output(BatchInsightResult) 사용 — LLM 1번 호출
    """
    docs = state.get("retrieved_docs", [])
    if not docs:
        return {"review_insights": {}}

    llm        = get_llm()
    skin_type  = state.get("skin_type", "")

    # 제품별로 리뷰 묶기
    product_docs: dict[str, list] = defaultdict(list)
    for doc in docs:
        goods_no = doc.metadata.get("goods_no", "unknown")
        product_docs[goods_no].append(doc)

    # 배치 프롬프트 구성
    batch_text = ""
    for goods_no, pdocs in product_docs.items():
        reviews_text = "\n".join(
            f"- {doc.page_content[:200]}" for doc in pdocs[:5]
        )
        batch_text += f"\n[{goods_no}]\n{reviews_text}\n"

    # structured_output으로 LLM 1번 호출
    structured_llm = llm.with_structured_output(BatchInsightResult)
    result: BatchInsightResult = structured_llm.invoke([
        {"role": "system", "content": REVIEW_INSIGHT_SYSTEM},
        {"role": "human",  "content": REVIEW_INSIGHT_HUMAN.format(
            skin_type=skin_type or "미입력",
            batch_text=batch_text,
        )},
    ])

    return {
        "review_insights": {
            item.goods_no: item.model_dump(exclude={"goods_no"})
            for item in result.insights
        },
        "retrieved_docs": docs,
    }


# ── 노드 8: generate_answer_node ─────────────────────────────────────────────

def generate_answer_node(state: GraphState) -> dict:
    """
    최종 답변 생성 — with_structured_output(GenerateResult) 사용
    - 비교: structured_compare_result + product_summaries (review_insight 스킵됨)
    - 추천/필터: review_insights + retrieved_docs
    """
    llm               = get_llm()
    query_type        = state.get("query_type", "일반")
    guide             = GENERATE_GUIDE.get(query_type, GENERATE_GUIDE["필터해석"])
    product_summaries = state.get("product_summaries", [])
    review_insights   = state.get("review_insights", {})
    structured_result = state.get("structured_compare_result", {})
    compare_axes      = state.get("compare_axes", [])

    if query_type == "비교" and product_summaries:
        compare_text = _format_compare_summaries(product_summaries)
        axes_info    = f"비교 기준: {', '.join(compare_axes)}\n\n" if compare_axes else ""
        if structured_result and "judgments" in structured_result:
            axis_text = _format_structured_result(structured_result["judgments"])
            docs_text = f"{axes_info}[축별 판정 결과]\n{axis_text}\n\n[제품별 상세 요약]\n{compare_text}"
        else:
            docs_text = axes_info + compare_text
    elif review_insights:
        insight_text = _format_insights(
            review_insights,
            skin_type=state.get("skin_type", ""),
            skin_tone=state.get("skin_tone", ""),
        )
        raw_text     = _format_docs(state.get("retrieved_docs", []))
        docs_text    = f"[제품별 리뷰 인사이트]\n{insight_text}\n\n[참고 리뷰]\n{raw_text}"
    else:
        docs_text = _format_docs(state.get("retrieved_docs", []))

    # skin_type / skin_tone → generate 프롬프트에 전달
    # LLM이 유저 피부 조건을 인식하고 주의 항목에서 모순 방지
    skin_type = state.get("skin_type", "") or "미입력"
    skin_tone = state.get("skin_tone", "") or "미입력"

    # structured_output으로 answer + recommended 동시 추출 (원칙 ② 준수)
    result: GenerateResult = llm.with_structured_output(GenerateResult).invoke([
        {"role": "system", "content": GENERATE_SYSTEM},
        {"role": "human",  "content": GENERATE_HUMAN.format(
            query_type=query_type,
            skin_type=skin_type,
            skin_tone=skin_tone,
            guide=guide,
            question=_get_last_human_message(state),
            docs=docs_text,
        )},
    ])

    existing = state.get("recommended_products", [])
    merged   = list(dict.fromkeys(existing + result.recommended))

    # mention 수치 + 평점을 답변에 직접 주입
    # 비교일 때는 product_summaries에서 goods_no 추출
    if query_type == "비교" and product_summaries:
        compare_insights = {}
        for s in product_summaries:
            d = s if isinstance(s, dict) else {}
            gno = d.get("goods_no", "")
            pname = d.get("product_name", "")
            if gno:
                # product_name을 함께 저장해서 답변에서 짧은 이름으로 검색 가능하게
                compare_insights[gno] = {"product_name": pname}
        answer = _inject_mention_stats(result.answer, compare_insights)
    else:
        answer = _inject_mention_stats(result.answer, review_insights)

    return {
        "messages":             [AIMessage(content=answer)],
        "recommended_products": merged,
    }


# ── 노드 9: rewrite_query_node ───────────────────────────────────────────────

def rewrite_query_node(state: GraphState) -> dict:
    llm      = get_llm()
    question = _get_last_human_message(state)
    response = llm.invoke([
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "human",  "content": REWRITE_HUMAN.format(question=question)},
    ])
    # rewrite → intake 재진입 시 skin_type/skin_tone/user_needs 보존
    return {
        "messages":      [HumanMessage(content=response.content.strip())],
        "rewrite_count": state.get("rewrite_count", 0) + 1,
        "skin_type":     state.get("skin_type", ""),
        "skin_tone":     state.get("skin_tone", ""),
        "user_needs":    state.get("user_needs", []),
    }


# ── 노드 10: generate_fallback_node ──────────────────────────────────────────

def generate_fallback_node(state: GraphState) -> dict:
    llm      = get_llm()
    question = _get_last_human_message(state)
    response = llm.invoke([
        {"role": "system", "content": FALLBACK_SYSTEM},
        {"role": "human",  "content": FALLBACK_HUMAN.format(question=question)},
    ])
    return {"messages": [AIMessage(content=response.content)]}


# ── 헬퍼 함수 ────────────────────────────────────────────────────────────────

def _get_last_human_message(state: GraphState) -> str:
    """messages에서 마지막 HumanMessage 내용 반환"""
    for msg in reversed(state["messages"]):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content
    return ""


def _match_skin_type(text: str) -> str:
    """키워드 매칭으로 피부타입 추출 (LLM 없음)"""
    for label in SKIN_TYPE_MAP:
        if label in text:
            return label
    return ""


def _match_skin_tone(text: str) -> str:
    """키워드 매칭으로 피부톤 추출 (LLM 없음)"""
    # 구어체 포함 매핑
    tone_keywords = {
        "여름쿨": ["여름쿨", "여쿨"],
        "봄웜":   ["봄웜"],
        "가을웜": ["가을웜", "갈웜"],
        "쿨톤":   ["쿨톤", "파란끼", "분홍빛"],
        "웜톤":   ["웜톤", "노란끼", "황톤", "노르스름"],
        "뉴트럴": ["뉴트럴"],
        "모름":   ["모름", "몰라", "모르겠"],
    }
    for tone, keywords in tone_keywords.items():
        for kw in keywords:
            if kw in text:
                # "모름"은 빈 문자열로 처리 (피부톤 없이 진행)
                return "" if tone == "모름" else tone
    return ""


def _format_docs(docs: list) -> str:
    if not docs:
        return "관련 리뷰를 찾지 못했습니다."
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        name = m.get('goods_name') or m.get('goods_no', '')
        parts.append(
            f"[리뷰 {i}] 제품: {name} | "
            f"평점: {m.get('score', '')} | "
            f"피부타입: {m.get('skin_type', '미입력')} | "
            f"재구매: {'예' if m.get('is_repurchase') else '아니오'}\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)


def _bar(score: float, length: int = 8) -> str:
    """0~1 스코어 → 텍스트 바 (예: ██████░░ 78%)"""
    filled = round(score * length)
    return "█" * filled + "░" * (length - filled) + f" {int(score * 100)}%"


def _clean_product_name(name: str) -> str:
    """[태그] 제거 후 핵심 제품명만 반환"""
    import re
    return re.sub(r"\[.*?\]", "", name).strip()


def _inject_mention_stats(answer: str, review_insights: dict) -> str:
    """LLM 답변에 mention 바 차트 + 헤더 평점/리뷰수 직접 주입"""
    if not review_insights:
        return answer
    try:
        from rag.retriever import STATS_PATH
        import json as _json
        if not STATS_PATH.exists():
            return answer
        ps = _json.loads(STATS_PATH.read_text(encoding="utf-8"))
        summary = load_review_summary()
        aspects_kr = {
            "non_sticky": "끈적임없음", "fresh": "산뜻함",
            "moisture": "보습력",       "mild": "자극없음",
            "no_whitecast": "백탁없음", "no_pilling": "밀림없음",
            "longevity": "지속력",      "no_eye_irritation": "눈시림없음",
        }
        injected = set()
        for goods_no, insight_data in review_insights.items():
            if goods_no in injected:
                continue
            item = summary.get(goods_no, {})
            raw_name = item.get("goods_name", "") if isinstance(item, dict) else ""
            if not raw_name:
                continue
            clean_name = _clean_product_name(raw_name)
            # 비교 모드에서는 product_name(짧은 이름)도 활용
            product_name = insight_data.get("product_name", "") if isinstance(insight_data, dict) else ""
            # mention 바 차트 계산
            s = ps.get(goods_no, {})
            total = s.get("total_reviews", 1) or 1
            m_items = []
            for asp, kr in aspects_kr.items():
                mention = s.get(asp + "_mention", 0) or 0
                pos     = s.get(asp + "_pos")
                if mention >= 5 and pos is not None:
                    m_items.append((mention / total, mention, pos, kr))
            m_items.sort(reverse=True)
            stats_block = ""
            if m_items:
                stat_lines = []
                for ratio, mention, pos, kr in m_items[:3]:
                    filled = int(pos * 8)
                    bar = "█" * filled + "░" * (8 - filled)
                    stat_lines.append(
                        "> " + bar + " **" + kr + "** " +
                        str(mention) + "건 언급 · 긍정률 " + str(int(pos*100)) + "%"
                    )
                stats_block = "\n".join(stat_lines) + "\n"
            # 메타 정보
            meta = item.get("metadata", {}) if isinstance(item, dict) else {}
            rating    = meta.get("stat_rating")
            rev_count = meta.get("stat_review_count")
            suffix = ""
            if rating:
                suffix += "  ⭐" + str(rating)
            if rev_count:
                suffix += " · " + format(rev_count, ",") + "개 리뷰"
            # ── 추천 헤더(1️⃣, 2️⃣...) 검색 ──
            found = False
            for search in [clean_name, clean_name[:12]]:
                for emoji in ["1️⃣", "2️⃣", "3️⃣", "4️⃣"]:
                    idx = answer.find(emoji)
                    while idx != -1:
                        eol = answer.find("\n", idx)
                        if eol == -1:
                            break
                        header_line = answer[idx:eol]
                        if search in header_line:
                            if suffix and str(rating or "") not in header_line:
                                answer = answer[:eol] + suffix + answer[eol:]
                                eol += len(suffix)
                            if stats_block:
                                answer = answer[:eol+1] + stats_block + answer[eol+1:]
                            found = True
                            break
                        idx = answer.find(emoji, idx + 1)
                    if found:
                        break
                if found:
                    break
            # ── 비교 모드: "이런 분께 추천해요" bullet 검색 ──
            if not found and stats_block:
                # product_name(짧은 이름) 또는 clean_name 앞부분으로 bullet 검색
                search_names = []
                if product_name:
                    search_names.append(product_name)
                search_names.append(clean_name[:6])
                for sname in search_names:
                    bullet = "- **" + sname
                    idx = answer.find(bullet)
                    if idx != -1:
                        eol = answer.find("\n", idx)
                        if eol != -1:
                            cmp_block = "\n".join(
                                "  " + line
                                for line in stats_block.strip().split("\n")[:2]
                            ) + "\n"
                            answer = answer[:eol+1] + cmp_block + answer[eol+1:]
                            found = True
                            break
            injected.add(goods_no)
    except Exception:
        pass
    return answer


def _is_contradictory(text: str, skin_type: str, skin_tone: str) -> bool:
    """
    failure_points 항목이 유저 조건과 모순되는지 확인

    1. 유저 피부타입 직접 매칭
       예) 유저=지성, text="지성 피부에는 유분감" → True

    2. 다른 피부타입 언급도 제거
       예) 유저=민감성, text="지성 피부에서 번들거림" → True
       (민감성 피부 추천인데 지성 피부 관련 주의는 의미없음)

    3. 유저 피부톤 직접 매칭
       예) 유저=여쿨, text="쿨톤에 어울리지 않을 수 있음" → True
    """
    ALL_SKIN_TYPES = ["지성", "건성", "복합성", "민감성", "중성", "복합건성", "복합지성", "수부지"]

    # 유저 피부타입 직접 매칭
    if skin_type and skin_type in text:
        return True

    # 유저 피부타입이 있으면 다른 피부타입 언급도 제거
    if skin_type:
        for other in ALL_SKIN_TYPES:
            if other != skin_type and other in text:
                return True

    # 유저 피부톤 직접 매칭 (동의어 포함)
    SKIN_TONE_SYNONYMS = {
        "쿨톤":   ["쿨톤"],
        "웜톤":   ["웜톤"],
        "여름쿨": ["여름쿨", "여쿨", "쿨톤"],
        "봄웜":   ["봄웜", "웜톤"],
        "가을웜": ["가을웜", "갈웜", "웜톤"],
        "뉴트럴": ["뉴트럴"],
    }
    if skin_tone:
        for kw in SKIN_TONE_SYNONYMS.get(skin_tone, [skin_tone]):
            if kw in text:
                return True

    return False


def _format_insights(insights: dict, skin_type: str = "", skin_tone: str = "") -> str:
    if not insights:
        return ""
    summary = load_review_summary()  # {goods_no: {goods_name, metadata}}
    lines = []
    for goods_no, data in insights.items():
        item   = summary.get(goods_no, {})
        # summary에 goods_name이 없으면 goods_no 그대로 사용
        if isinstance(item, dict) and "metadata" in item:
            meta = item["metadata"]
            raw_name = item.get("goods_name", goods_no)
        else:
            # load_review_summary가 metadata dict만 반환하는 경우 대비
            meta = item
            raw_name = goods_no

        name      = _clean_product_name(raw_name)
        rating    = meta.get("stat_rating")
        rev_count = meta.get("stat_review_count")
        spread    = meta.get("spreadability_score")
        mild      = meta.get("mildness_score")

        # 헤더: 제품명 + 평점 + 리뷰수
        header = f"**{name}**"
        if rating:
            header += f"  ⭐ {rating}"
        if rev_count:
            header += f" · 리뷰 {rev_count:,}개"
        lines.append(header)

        # 발림성 / 순함 바 차트
        if spread is not None:
            lines.append(f"  발림성 {_bar(spread)}")
        if mild is not None:
            lines.append(f"  순함   {_bar(mild)}")

        # mention 수치 — RAG Grounding 근거 표시
        # product_stats에서 가장 많이 언급된 aspect top2 표시
        try:
            from rag.retriever import STATS_PATH
            import json as _json
            if STATS_PATH.exists():
                _ps = _json.loads(STATS_PATH.read_text(encoding="utf-8"))
                _s  = _ps.get(goods_no, {})
                _total = _s.get("total_reviews", 1) or 1
                _aspects_kr = {
                    "non_sticky": "끈적임없음", "fresh": "산뜻함",
                    "moisture": "보습력", "mild": "자극없음",
                    "no_whitecast": "백탁없음", "no_pilling": "밀림없음",
                    "longevity": "지속력", "no_eye_irritation": "눈시림없음",
                }
                _mention_items = []
                for asp, kr in _aspects_kr.items():
                    mention = _s.get(f"{asp}_mention", 0) or 0
                    pos     = _s.get(f"{asp}_pos")
                    if mention >= 5 and pos is not None:
                        ratio = mention / _total
                        _mention_items.append((ratio, mention, pos, kr))
                _mention_items.sort(reverse=True)
                if _mention_items:
                    top2 = _mention_items[:2]
                    stats_str = "  📊 " + " · ".join(
                        f"{kr} 언급 {mention}건({ratio:.0%}) 긍정률 {pos:.0%}"
                        for ratio, mention, pos, kr in top2
                    )
                    lines.append(stats_str)
        except Exception:
            pass

        # 인사이트
        if data.get("positive_keywords"):
            lines.append(f"  👍 {', '.join(data['positive_keywords'])}")
        if data.get("negative_keywords"):
            lines.append(f"  👎 {', '.join(data['negative_keywords'])}")
        if data.get("failure_points"):
            # 유저 피부타입/피부톤과 모순되는 항목 코드 레벨에서 필터링
            filtered_fp = [
                fp for fp in data["failure_points"]
                if not _is_contradictory(fp, skin_type, skin_tone)
            ]
            if filtered_fp:
                lines.append(f"  ⚠️  {', '.join(filtered_fp)}")
        if data.get("overall_sentiment"):
            lines.append(f"  전반적 반응: {data['overall_sentiment']}")
        lines.append("")  # 제품 간 빈 줄
    return "\n".join(lines)


def _format_compare_summaries(summaries: list) -> str:
    parts = []
    for s in summaries:
        try:
            data  = json.loads(s) if isinstance(s, str) else s
            name  = data.get("product_name", "")
            axes  = data.get("axes", [])   # list[{axis, analysis}]
            best  = data.get("best_for", "")
            avoid = data.get("avoid_if", "")
            fmt_lines = [f"[{name}]"]
            for item in axes:
                if isinstance(item, dict):
                    fmt_lines.append(f"  {item.get('axis', '')}: {item.get('analysis', '')}")
            lines = fmt_lines
            if best:
                lines.append(f"  추천 대상: {best}")
            if avoid:
                lines.append(f"  주의 대상: {avoid}")
            parts.append("\n".join(fmt_lines))
        except Exception:
            parts.append(str(s))
    return "\n\n".join(parts)


def _format_structured_result(judgments: list) -> str:
    """judgments: list[{axis, winner, draw, reason}]"""
    result_lines = []
    for j in judgments:
        axis = j.get("axis", "")
        if j.get("draw"):
            result_lines.append(f"  {axis}: 무승부 — {j.get('reason', '')}")
        else:
            result_lines.append(f"  {axis}: {j.get('winner', '')} 우세 — {j.get('reason', '')}")
    return "\n".join(result_lines)