"""
graph/subgraphs/filter.py

fil_subgraph: 필터해석 / follow_up / refine / 일반 처리

기획 의도:
    올리브영에서 피부타입·피부톤 필터를 누르고 리뷰를 일일이 읽어야 하는
    불편함을 해결. "지성 피부 유저들이 이 제품 어떻게 봤어?" 한 마디로
    총평을 받는 기능이 핵심.

    follow_up: 이전 추천 결과에 대한 후속 질문
               → selected_products / recommended_products 컨텍스트 유지
    refine:    조건 변경 재추천 → 라우터에서 rec_sub로 보내므로 여기 안 옴
    일반:      피부타입/피부톤 없으면 general_search만 실행

검색 전략:
    1. 피부타입 있으면 search_by_skin_type (k=30)
    2. 피부톤 있으면 search_by_skin_tone (k=30)
    3. 제품 언급 있으면 search_by_goods_no (k=30)  ← 필터해석 핵심
    4. follow_up이면 이전 selected/recommended 제품도 검색
    5. 위 결과 없으면 general_search fallback (k=30)
    → 전체 dedup 후 retrieved_docs 저장
"""

import logging
from langchain_core.documents import Document

from graph.state import GraphState
from rag.retriever import (
    SKIN_TYPE_MAP,
    SKIN_TONE_MAP,
    general_search,
    search_by_skin_type,
    search_by_skin_tone,
    search_by_goods_no,
    resolve_goods_no,
)

logger = logging.getLogger(__name__)


def run_filter_subgraph(state: GraphState) -> dict:
    query        = _get_query(state)
    query_type   = state.get("query_type", "일반")
    skin_type    = state.get("skin_type", "")
    skin_tone    = state.get("skin_tone", "")
    products     = state.get("products_mentioned", [])

    skin_type_code = SKIN_TYPE_MAP.get(skin_type, "")
    skin_tone_code = SKIN_TONE_MAP.get(skin_tone, "")

    all_docs: list[Document] = []
    seen_ids: set[str] = set()

    def add_docs(docs: list[Document]) -> None:
        for doc in docs:
            doc_id = doc.metadata.get("id", doc.page_content[:50])
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_docs.append(doc)

    # 1. 피부타입 필터 검색 (필터해석 핵심)
    if skin_type_code:
        try:
            add_docs(search_by_skin_type(query, skin_type_code, k=30))
        except Exception as e:
            logger.warning("search_by_skin_type 실패: %s", e)

    # 2. 피부톤 필터 검색
    if skin_tone_code:
        try:
            add_docs(search_by_skin_tone(query, skin_tone_code, k=30))
        except Exception as e:
            logger.warning("search_by_skin_tone 실패: %s", e)

    # 3. 언급된 제품 검색 (특정 제품에 대한 필터해석)
    for product_name in products:
        goods_no = resolve_goods_no(product_name)
        if not goods_no:
            continue
        try:
            add_docs(search_by_goods_no(query, goods_no, skin_type_code, k=30))
        except Exception as e:
            logger.warning("search_by_goods_no(%s) 실패: %s", goods_no, e)

    # 4. follow_up: 이전 대화의 추천/선택 제품도 검색
    if query_type == "follow_up":
        prev_products = (
            state.get("selected_products", []) +
            state.get("recommended_products", [])
        )
        for product_name in prev_products:
            goods_no = resolve_goods_no(product_name)
            if not goods_no:
                continue
            try:
                add_docs(search_by_goods_no(query, goods_no, skin_type_code, k=10))
            except Exception as e:
                logger.warning("follow_up search(%s) 실패: %s", goods_no, e)

    # 5. 결과 없으면 general_search fallback
    if not all_docs:
        try:
            add_docs(general_search(query, k=30))
        except Exception as e:
            logger.warning("general_search fallback 실패: %s", e)

    return {"retrieved_docs": all_docs}


def _get_query(state: GraphState) -> str:
    for msg in reversed(state["messages"]):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content
    return ""