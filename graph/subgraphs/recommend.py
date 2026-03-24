"""
graph/subgraphs/recommend.py
추천 서브그래프 (pgvector 기반)

4가지 경우의 수:
    1. 피부타입 O / 피부톤 X
       → product_stats 스코어링 + skin_type 필터 샘플링

    2. 피부타입 O / 피부톤 O
       → product_stats 스코어링 + skin_type + skin_tone 필터 샘플링

    3. 피부타입 X / 피부톤 X
       → intake에서 needs_skin_type=True → interrupt (ask_skin)

    4. 피부타입 X / 피부톤 O  ← 핵심 개선
       → product_stats 스코어링 스킵
       → skin_tone 기반 제품별 균등 샘플링 직접 수행
       → 해당 피부톤 유저 리뷰에서 query와 유사한 제품 추출
"""

from graph.state import GraphState
from rag.retriever import (
    search_by_product_stats,
    search_balanced_by_products,
    search_by_skin_tone,
    normalize_skin_type,
    SKIN_TYPE_MAP,
    SKIN_TONE_MAP,
)


def _get_query(state: GraphState) -> str:
    """messages에서 마지막 HumanMessage 내용 반환"""
    for msg in reversed(state["messages"]):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content
    return ""


def run_recommend_subgraph(state: GraphState) -> dict:
    skin_type  = state.get("skin_type", "")
    skin_tone  = state.get("skin_tone", "")
    query      = _get_query(state)
    user_needs = state.get("user_needs", [])

    skin_type_normalized = normalize_skin_type(skin_type)
    skin_type_code       = SKIN_TYPE_MAP.get(skin_type, "")
    skin_tone_code       = SKIN_TONE_MAP.get(skin_tone, "")

    # ── 케이스 4: 피부타입 X, 피부톤 O ──────────────────────────────────────
    # "여쿨인데 톤업 선크림" → 피부톤 유저 리뷰 기반 직접 검색
    # product_stats 스코어링(피부타입 기반)을 쓰면 피부톤 정보가 무시됨
    if not skin_type and skin_tone:
        docs = search_by_skin_tone(
            query=query,
            skin_tone_code=skin_tone_code,
            skin_type_code="",  # 피부타입 없음
        )
        return {"retrieved_docs": docs}

    # ── 케이스 1, 2: 피부타입 O ──────────────────────────────────────────────
    # product_stats 스코어링 → 상위 5개 제품 선정
    # skin_tone_code: 피부타입 0.6 + 피부톤 매칭 0.4 블렌딩
    # already_recommended: 이전 추천 제품 패널티 (다양성 확보)
    already_recommended = state.get("recommended_products", [])
    top_goods_nos = search_by_product_stats(
        user_needs=user_needs,
        skin_type=skin_type_normalized,
        skin_tone_code=skin_tone_code,
        already_recommended=already_recommended,
    )

    if not top_goods_nos:
        return {"retrieved_docs": []}

    # 제품별 균등 샘플링
    # 케이스 1: skin_tone_code="" → skin_type만 필터
    # 케이스 2: skin_tone_code 있음 → skin_type + skin_tone 필터
    docs = search_balanced_by_products(
        query=query,
        goods_nos=top_goods_nos,
        skin_type_code=skin_type_code,
        skin_tone_code=skin_tone_code,
    )

    return {"retrieved_docs": docs}