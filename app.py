"""
app.py
langgraph dev 전용 진입점

실행:
    langgraph dev

접속:
    LangGraph Studio → https://smith.langchain.com/studio (로컬 서버 자동 연결)
    또는 http://localhost:2024

세션 정책:
    - langgraph dev가 자체 체크포인터를 주입하므로 checkpointer=None으로 빌드
    - 서버 재시작 시 이전 대화 기록 초기화 (InMemory)
    - LangGraph Studio에서 New Thread 버튼으로 언제든 새 세션 시작 가능

환경변수 (.env):
    OPENAI_API_KEY=sk-...
    PG_HOST=localhost
    PG_PORT=5432
    PG_DBNAME=oliveyoung
    PG_USER=postgres
    PG_PASSWORD=...
"""

from graph.main_graph import build_graph
from rag.retriever import get_embeddings, load_review_summary, load_product_stats

# ── 서버 시작 시 워밍업 ──────────────────────────────────────────────────────
# 첫 요청에서 임베딩 모델/통계 로딩으로 인한 지연(~25초) 방지
# langgraph dev 시작 시 미리 로딩해둠
try:
    get_embeddings()          # ko-sroberta-multitask 모델 캐싱
    load_review_summary()     # DB products(올리브영 공식 통계) 캐싱
    load_product_stats()      # DB product_stats 캐싱
    print("✅ 워밍업 완료 (임베딩 모델 + DB 통계 캐시)")
except Exception as e:
    print(f"⚠️ 워밍업 실패 (첫 요청에서 자동 로딩): {e}")

# langgraph dev가 자체 체크포인터를 주입
# → checkpointer=None으로 빌드 (MemorySaver 중복 방지)
graph = build_graph(checkpointer=None)