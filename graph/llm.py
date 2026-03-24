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
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _llm