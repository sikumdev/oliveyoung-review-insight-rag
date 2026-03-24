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
 
# langgraph dev가 자체 체크포인터를 주입
# → checkpointer=None으로 빌드 (MemorySaver 중복 방지)
graph = build_graph(checkpointer=None)