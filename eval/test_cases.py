"""
eval/test_cases.py

올리브영 선크림 RAG 챗봇 평가 테스트셋

평가 지표:
    Hit@K: 상위 K개 추천 결과 중 expected_goods_nos에 포함된 제품이 1개 이상 있으면 1점
    MRR  : Mean Reciprocal Rank — 첫 번째 정답 제품의 순위 역수 평균

실행:
    python eval/run_eval.py

테스트 케이스 설계 기준:
    - product_stats 실제 데이터 기반으로 각 조건의 상위 제품 확인
    - 피부타입별 / user_needs별 / 피부톤별 / 멀티조건 포함
    - 총 20개 케이스
"""

# ── 제품 코드 상수 ──────────────────────────────────────────────────────────
PULLY         = "A000000240562"   # 풀리 쌀 세라 수분 선크림
MEDIHEAL      = "A000000232672"   # 메디힐 마데카소사이드 수분 선세럼
COSRX         = "A000000211309"   # 코스알엑스 울트라 라이트 물막 수분 선세럼
REALBARRIER   = "A000000200673"   # 리얼베리어 세라 수분 장벽 선크림
ROUNDLAB_MOI  = "A000000246780"   # 라운드랩 자작나무 수분 선크림
ROUNDLAB_TONE = "A000000186166"   # 라운드랩 자작나무 수분 톤업 선크림
DALBA_PINK    = "A000000232725"   # 달바 핑크 톤업 선크림
DALBA_PURPLE  = "A000000200614"   # 달바 퍼플 톤업 선크림
GUDAL         = "A000000219553"   # 구달 맑은 어성초 진정 수분 선크림
DOCTORG       = "A000000180162"   # 닥터지 그린 마일드 업 선
SKINFOOD      = "A000000151904"   # 식물나라 워터프루프 선크림
ESTLA         = "A000000219094"   # 에스트라 더마UV365 비타C 선크림
SKIN1004      = "A000000215559"   # 스킨1004 센텔라 워터핏 선세럼
ANUA          = "A000000207901"   # 아누아 어성초 실키 모이스처 선크림
UUIIK         = "A000000227070"   # 유이크 톤업 선크림 보라
MEDIHEAL2     = "A000000202359"   # 메이크프렘 수딩 핑크 톤업 선크림
ZEROED        = "A000000209917"   # 제로이드 데일리 선크림


# ── 테스트 케이스 ───────────────────────────────────────────────────────────
TEST_CASES = [

    # ── 추천: 피부타입 단독 ────────────────────────────────────────────────
    {
        "id": "rec_001",
        "category": "추천/피부타입",
        "query": "지성 피부인데 선크림 추천해줘",
        "skin_type": "지성",
        "skin_tone": "",
        "user_needs": [],
        "expected_goods_nos": [PULLY, MEDIHEAL, COSRX],
        "description": "지성 피부 기본 추천 — non_sticky + fresh 가중치 높음",
    },
    {
        "id": "rec_002",
        "category": "추천/피부타입",
        "query": "건성 피부인데 촉촉한 선크림 추천해줘",
        "skin_type": "건성",
        "skin_tone": "",
        "user_needs": ["보습"],
        "expected_goods_nos": [PULLY, REALBARRIER, MEDIHEAL],
        "description": "건성+보습 — moisture_pos 가중치 높음",
    },
    {
        "id": "rec_003",
        "category": "추천/피부타입",
        "query": "민감성 피부인데 자극 없는 선크림 추천해줘",
        "skin_type": "민감성",
        "skin_tone": "",
        "user_needs": ["자극없음"],
        "expected_goods_nos": [PULLY, REALBARRIER, ESTLA],
        "description": "민감성+자극없음 — mild_pos 가중치 높음",
    },
    {
        "id": "rec_004",
        "category": "추천/피부타입",
        "query": "복합성 피부인데 화장 밀리지 않는 선크림 추천해줘",
        "skin_type": "복합성",
        "skin_tone": "",
        "user_needs": ["밀림없음"],
        "expected_goods_nos": [PULLY, MEDIHEAL, COSRX],
        "description": "복합성+밀림없음 — no_pilling_pos 가중치 높음",
    },

    # ── 추천: user_needs 복합 ──────────────────────────────────────────────
    {
        "id": "rec_005",
        "category": "추천/복합니즈",
        "query": "지성인데 끈적임 없고 화장도 잘 먹는 선크림",
        "skin_type": "지성",
        "skin_tone": "",
        "user_needs": ["끈적임없음", "밀림없음"],
        "expected_goods_nos": [PULLY, MEDIHEAL, COSRX],
        "description": "지성+끈적임없음+밀림없음",
    },
    {
        "id": "rec_006",
        "category": "추천/복합니즈",
        "query": "민감성인데 눈시림도 없고 순한 선크림 추천해줘",
        "skin_type": "민감성",
        "skin_tone": "",
        "user_needs": ["눈시림없음", "자극없음"],
        "expected_goods_nos": [REALBARRIER, ESTLA, MEDIHEAL],
        "description": "민감성+눈시림없음+자극없음",
    },
    {
        "id": "rec_007",
        "category": "추천/복합니즈",
        "query": "건성인데 오래 가는 보습 선크림",
        "skin_type": "건성",
        "skin_tone": "",
        "user_needs": ["보습", "지속력"],
        "expected_goods_nos": [PULLY, REALBARRIER, ANUA],
        "description": "건성+보습+지속력",
    },

    # ── 추천: 피부톤 단독 (케이스4) ────────────────────────────────────────
    {
        "id": "rec_008",
        "category": "추천/피부톤",
        "query": "여쿨인데 톤업 선크림 추천해줘",
        "skin_type": "",
        "skin_tone": "여름쿨",
        "user_needs": [],
        "expected_goods_nos": [DALBA_PURPLE, ROUNDLAB_TONE, UUIIK],
        "description": "여름쿨 피부톤 — B05 코드 리뷰 기반 검색",
    },
    {
        "id": "rec_009",
        "category": "추천/피부톤",
        "query": "가을웜인데 자연스러운 톤업 선크림 있어?",
        "skin_type": "",
        "skin_tone": "가을웜",
        "user_needs": [],
        "expected_goods_nos": [MEDIHEAL2, ESTLA, DALBA_PINK],
        "description": "가을웜 피부톤 — B06 코드 리뷰 기반 검색",
    },

    # ── 추천: 피부타입 + 피부톤 (케이스2) ─────────────────────────────────
    {
        "id": "rec_010",
        "category": "추천/피부타입+피부톤",
        "query": "지성이고 여쿨인데 산뜻한 선크림 추천해줘",
        "skin_type": "지성",
        "skin_tone": "여름쿨",
        "user_needs": ["산뜻함"],
        "expected_goods_nos": [PULLY, COSRX, MEDIHEAL],
        "description": "지성+여름쿨+산뜻함 — 블렌딩 스코어링",
    },
    {
        "id": "rec_011",
        "category": "추천/피부타입+피부톤",
        "query": "민감성이고 쿨톤인데 자극 없는 선크림",
        "skin_type": "민감성",
        "skin_tone": "쿨톤",
        "user_needs": ["자극없음"],
        "expected_goods_nos": [PULLY, REALBARRIER, ESTLA],
        "description": "민감성+쿨톤+자극없음",
    },

    # ── 필터해석 ───────────────────────────────────────────────────────────
    {
        "id": "fil_001",
        "category": "필터해석/피부타입",
        "query": "지성 피부 유저들이 라운드랩 어떻게 생각해?",
        "skin_type": "지성",
        "skin_tone": "",
        "user_needs": [],
        "products_mentioned": [ROUNDLAB_MOI],
        "expected_goods_nos": [ROUNDLAB_MOI],
        "description": "지성+특정제품 필터해석",
    },
    {
        "id": "fil_002",
        "category": "필터해석/피부톤",
        "query": "여쿨인데 달바 퍼플 어때?",
        "skin_type": "",
        "skin_tone": "여름쿨",
        "user_needs": [],
        "products_mentioned": [DALBA_PURPLE],
        "expected_goods_nos": [DALBA_PURPLE],
        "description": "여름쿨+특정제품 필터해석",
    },
    {
        "id": "fil_003",
        "category": "필터해석/피부타입",
        "query": "건성 피부인데 구달 선크림 어때?",
        "skin_type": "건성",
        "skin_tone": "",
        "user_needs": [],
        "products_mentioned": [GUDAL],
        "expected_goods_nos": [GUDAL],
        "description": "건성+특정제품 필터해석",
    },

    # ── 비교 ───────────────────────────────────────────────────────────────
    {
        "id": "cmp_001",
        "category": "비교",
        "query": "달바랑 라운드랩 비교해줘",
        "skin_type": "",
        "skin_tone": "",
        "user_needs": [],
        "products_mentioned": [DALBA_PINK, ROUNDLAB_MOI],
        "expected_goods_nos": [DALBA_PINK, ROUNDLAB_MOI],
        "description": "2개 제품 비교",
    },
    {
        "id": "cmp_002",
        "category": "비교",
        "query": "라운드랩이랑 구달이랑 닥터지 비교해줘",
        "skin_type": "",
        "skin_tone": "",
        "user_needs": [],
        "products_mentioned": [ROUNDLAB_MOI, GUDAL, DOCTORG],
        "expected_goods_nos": [ROUNDLAB_MOI, GUDAL, DOCTORG],
        "description": "3개 제품 비교",
    },

    # ── 엣지케이스 ─────────────────────────────────────────────────────────
    {
        "id": "edge_001",
        "category": "엣지케이스",
        "query": "선크림 뭐가 좋냐",
        "skin_type": "복합성",    # interrupt 후 입력된 피부타입
        "skin_tone": "",
        "user_needs": [],
        "expected_goods_nos": [PULLY, MEDIHEAL, COSRX],
        "description": "반말 + 정보 없음 → interrupt 후 복합성 기준",
    },
    {
        "id": "edge_002",
        "category": "엣지케이스",
        "query": "recommend sunscreen for oily skin",
        "skin_type": "지성",
        "skin_tone": "",
        "user_needs": [],
        "expected_goods_nos": [PULLY, MEDIHEAL, COSRX],
        "description": "영어 입력 → 지성 기준 추천",
    },
    {
        "id": "edge_003",
        "category": "엣지케이스",
        "query": "여드름성 피부인데 자극없는 선크림 추천해줘",
        "skin_type": "민감성",   # 여드름성 → 민감성으로 처리
        "skin_tone": "",
        "user_needs": ["자극없음"],
        "expected_goods_nos": [PULLY, REALBARRIER, ESTLA],
        "description": "여드름성 → 민감성 매핑",
    },
    {
        "id": "edge_004",
        "category": "엣지케이스",
        "query": "노란끼 많은데 자극 없는 선크림 추천해줘",
        "skin_type": "",
        "skin_tone": "웜톤",     # 노란끼 → 웜톤 매핑
        "user_needs": ["자극없음"],
        "expected_goods_nos": [PULLY, MEDIHEAL, "A000000224140"],  # 실측 기반 수정
        "description": "노란끼 → 웜톤 구어체 매핑 (웜톤 리뷰 기반 실측값)",
    },
]