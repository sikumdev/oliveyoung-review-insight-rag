"""
graph/routers.py

LangGraph 라우터 함수 모음 (순수 로직 — LLM 호출 없음)

라우터 목록:
    route_question_router      intake_node 이후 서브그래프 분기
    validate_router            검색 결과 충분 여부 → 다음 단계 결정
    after_ask_skin_type_router 피부타입 입력 후 원래 query_type으로 재분기
    after_ask_compare_router   비교 제품 입력 후 제품 수 확인

라우팅 전체 흐름:

    START
      │
    intake_node  (LLM — IntakeResult structured_output)
      │
    route_question_router
      ├─ needs_skin_type == True          → ask_skin
      │     └─ after_ask_skin_type_router → rec_sub / cmp_sub / ask_compare / fil_sub
      ├─ 추천 / refine                    → rec_sub
      ├─ 비교 + 제품 2개+                 → cmp_sub
      ├─ 비교 + 제품 부족                 → ask_compare
      │     └─ after_ask_compare_router  → cmp_sub (2개 모이면) / ask_compare (루프)
      └─ 필터해석 / follow_up            → fil_sub
           │
         validate_router
           ├─ 비교: product_summaries 있음  → generate  (review_insight 스킵)
           ├─ 추천/refine: docs >= 3        → review_insight → generate
           ├─ 필터해석: docs >= 3           → review_insight → generate
           ├─ follow_up: docs >= 1          → review_insight → generate
           ├─ 부족 & rewrite < 3            → rewrite → intake (재진입)
           └─ 부족 & rewrite >= 3           → fallback
"""

from graph.state import GraphState


# ── 라우터 1: route_question_router ──────────────────────────────────────────

def route_question_router(state: GraphState) -> str:
    """
    intake_node 이후 분기 결정

    우선순위:
    ① needs_skin_type == True           → ask_skin_type
    ② query_type == "추천" | "refine"   → rec_subgraph
    ③ query_type == "비교" + 제품 2개+  → cmp_subgraph
    ④ query_type == "비교" + 제품 부족  → ask_compare
    ⑤ 필터해석 / follow_up              → fil_subgraph
    """
    needs_skin_type = state.get("needs_skin_type", False)
    query_type      = state.get("query_type", "필터해석")
    products        = state.get("products_mentioned", [])

    if needs_skin_type:
        return "ask_skin_type"

    if query_type in ("추천", "refine"):
        return "rec_subgraph"

    if query_type == "비교":
        return "cmp_subgraph" if len(products) >= 2 else "ask_compare"

    # 필터해석 / follow_up → fil_subgraph
    return "fil_subgraph"


# ── 라우터 2: validate_router ─────────────────────────────────────────────────

def validate_router(state: GraphState) -> str:
    """
    서브그래프 실행 후 결과 충분 여부 확인

    충분 기준:
    - 비교:           product_summaries 있음
                      → generate 직행 (cmp_subgraph에서 요약 완료, review_insight 중복 방지)
    - 추천 / refine:  docs >= 3
    - 필터해석:       docs >= 3  (피부타입/피부톤 조건의 리뷰 충분히 필요)
    - follow_up:      docs >= 1  (이전 맥락 기반이라 적은 수로도 답변 가능)

    부족 처리:
    - rewrite_count < 3  → rewrite (질문 재작성 후 intake 재진입)
    - rewrite_count >= 3 → fallback (LLM 자체 지식으로 답변)
    """
    query_type        = state.get("query_type", "필터해석")
    docs              = state.get("retrieved_docs", [])
    product_summaries = state.get("product_summaries", [])
    rewrite_count     = state.get("rewrite_count", 0)

    # ── 비교: product_summaries 기준 ─────────────────────────────────────────
    if query_type == "비교":
        if bool(product_summaries):
            return "generate"      # review_insight 스킵
        if rewrite_count >= 3:
            return "fallback"
        return "rewrite"

    # ── 나머지: docs 수 기준 ──────────────────────────────────────────────────
    if query_type in ("추천", "refine", "필터해석"):
        sufficient = len(docs) >= 3
    else:
        # follow_up
        sufficient = len(docs) >= 1

    if sufficient:
        return "review_insight"

    if rewrite_count >= 3:
        return "fallback"

    return "rewrite"


# ── 라우터 3: after_ask_skin_type_router ─────────────────────────────────────

def after_ask_skin_type_router(state: GraphState) -> str:
    """
    ask_skin_type_node 이후 원래 query_type으로 재분기

    피부타입을 받은 뒤 중단된 흐름을 이어받아
    원래 의도한 서브그래프로 재진입
    """
    query_type = state.get("query_type", "필터해석")
    products   = state.get("products_mentioned", [])

    if query_type in ("추천", "refine"):
        return "rec_subgraph"

    if query_type == "비교":
        return "cmp_subgraph" if len(products) >= 2 else "ask_compare"

    return "fil_subgraph"


# ── 라우터 4: after_ask_compare_router ───────────────────────────────────────

def after_ask_compare_router(state: GraphState) -> str:
    """
    ask_compare_product_node 이후 제품 수 확인

    2개 이상 모이면 cmp_subgraph로, 아직 부족하면 다시 ask_compare
    (interrupt 루프 — 최대 2개 제품이 모일 때까지 반복)
    """
    products = state.get("products_mentioned", [])
    return "cmp_subgraph" if len(products) >= 2 else "ask_compare"