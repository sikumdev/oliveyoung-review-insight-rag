"""
graph/llm.py
LLM 싱글톤 — 순환 임포트 방지를 위해 nodes.py에서 분리

사용:
    from graph.llm import get_llm
"""

from langchain_openai import ChatOpenAI

_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        # tags=["nostream"]: 노드 안의 LLM 호출(구조화 출력 JSON 등)이 Studio 대화창에
        # 토큰 스트림으로 노출되지 않게 함. 최종 답변은 state의 messages로 표시됨.
        # (2026-11: langgraph-api 0.19로 올린 뒤 내부 JSON이 대화창에 보여서 추가)
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, tags=["nostream"])
    return _llm