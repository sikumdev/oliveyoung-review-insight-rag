"""
graph/main_graph.py

올리브영 선크림 RAG 챗봇 메인 그래프
라우터 로직은 graph/routers.py에서 관리
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import GraphState
from graph.routers import (
    route_question_router,
    validate_router,
    after_ask_skin_type_router,
    after_ask_compare_router,
)
from graph.nodes import (
    intake_node,
    ask_skin_type_node,
    ask_compare_product_node,
    rec_subgraph_node,
    cmp_subgraph_node,
    fil_subgraph_node,
    review_insight_node,
    generate_answer_node,
    rewrite_query_node,
    generate_fallback_node,
)


# ── 그래프 빌드 ───────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    builder = StateGraph(GraphState)

    builder.add_node("intake",      intake_node)
    builder.add_node("ask_skin",    ask_skin_type_node)
    builder.add_node("ask_compare", ask_compare_product_node)
    builder.add_node("rec_sub",     rec_subgraph_node)
    builder.add_node("cmp_sub",     cmp_subgraph_node)
    builder.add_node("fil_sub",     fil_subgraph_node)
    builder.add_node("review_ins",  review_insight_node)
    builder.add_node("generate",    generate_answer_node)
    builder.add_node("rewrite",     rewrite_query_node)
    builder.add_node("fallback",    generate_fallback_node)

    builder.add_edge(START, "intake")

    builder.add_conditional_edges(
        "intake",
        route_question_router,
        {
            "ask_skin_type": "ask_skin",
            "rec_subgraph":  "rec_sub",
            "cmp_subgraph":  "cmp_sub",
            "ask_compare":   "ask_compare",
            "fil_subgraph":  "fil_sub",
        },
    )

    builder.add_conditional_edges(
        "ask_skin",
        after_ask_skin_type_router,
        {
            "rec_subgraph": "rec_sub",
            "cmp_subgraph": "cmp_sub",
            "ask_compare":  "ask_compare",
            "fil_subgraph": "fil_sub",
        },
    )

    builder.add_conditional_edges(
        "ask_compare",
        after_ask_compare_router,
        {
            "cmp_subgraph": "cmp_sub",
            "ask_compare":  "ask_compare",
        },
    )

    for sub_node in ("rec_sub", "cmp_sub", "fil_sub"):
        builder.add_conditional_edges(
            sub_node,
            validate_router,
            {
                "review_insight": "review_ins",
                "generate":       "generate",   # 비교: review_insight 스킵
                "rewrite":        "rewrite",
                "fallback":       "fallback",
            },
        )

    builder.add_edge("rewrite",    "intake")
    builder.add_edge("review_ins", "generate")
    builder.add_edge("generate",   END)
    builder.add_edge("fallback",   END)

    return builder.compile(checkpointer=checkpointer)


# ── 싱글톤 그래프 인스턴스 ────────────────────────────────────────────────────

_graph = None


def get_graph(use_memory: bool = True):
    """
    싱글톤 그래프 반환
    use_memory=True  → MemorySaver 체크포인터 (로컬/테스트용)
    use_memory=False → 체크포인터 없음 (langgraph dev 자동 주입 환경)
    """
    global _graph
    if _graph is None:
        checkpointer = MemorySaver() if use_memory else None
        _graph = build_graph(checkpointer=checkpointer)
    return _graph