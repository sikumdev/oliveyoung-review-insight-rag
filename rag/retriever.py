"""
rag/retriever.py
pgvector 기반 검색 함수
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

RETRIEVE_K = 30
PRODUCT_K  = 5
SAMPLE_K   = 5

EMBED_MODEL  = "jhgan/ko-sroberta-multitask"
BASE_DIR            = Path(__file__).resolve().parent.parent
WEIGHTS_PATH        = BASE_DIR / "data" / "skin_type_weights.json"
REVIEW_SUMMARY_PATH = BASE_DIR / "data" / "review_summary.json"
STATS_PATH          = BASE_DIR / "data" / "product_stats.json"

SKIN_TYPE_MAP = {
    "지성":    "A01",
    "건성":    "A02",
    "복합성":  "A03",
    "복합":    "A03",
    "민감성":  "A04",
    "민감":    "A04",
    "중성":    "A05",
    "복합건성": "A06",
    "복합지성": "A07",
}

SKIN_TYPE_NORMALIZE = {"복합": "복합성", "민감": "민감성"}

def normalize_skin_type(skin_type: str) -> str:
    return SKIN_TYPE_NORMALIZE.get(skin_type, skin_type)

SKIN_TONE_MAP = {
    "쿨톤":    "B01",
    "웜톤":    "B02",
    "뉴트럴톤": "B03",
    "뉴트럴":  "B03",
    "봄웜":    "B04",
    "여름쿨":  "B05",
    "여쿨":    "B05",
    "가을웜":  "B06",
    "갈웜":    "B06",
}

NEEDS_TO_STAT = {
    "산뜻함":    "fresh_pos",
    "끈적임없음": "non_sticky_pos",
    "보습":      "moisture_pos",
    "자극없음":  "mild_pos",
    "백탁없음":  "no_whitecast_pos",
    "밀림없음":  "no_pilling_pos",
    "지속력":    "longevity_pos",
    "눈시림없음": "no_eye_irritation_pos",
}

AXES_TO_ASPECT: dict[str, str | None] = {
    "보습력":  "moisture",
    "보습":    "moisture",
    "촉촉함":  "moisture",
    "끈적임":  "non_sticky",
    "산뜻함":  "fresh",
    "발림성":  "no_pilling",
    "흡수력":  "no_pilling",
    "자극":    "mild",
    "자극없음": "mild",
    "백탁":    "no_whitecast",
    "지속력":  "longevity",
    "눈시림":  "no_eye_irritation",
    "눈시림없음": "no_eye_irritation",
    "밀림":    "no_pilling",
    "밀림없음": "no_pilling",
    "백탁없음": "no_whitecast",
    "향":      None,
}

_VALID_ASPECTS    = frozenset(["fresh","non_sticky","moisture","mild","no_whitecast","no_pilling","longevity","no_eye_irritation"])
_VALID_SENTIMENTS = frozenset(["positive","negative","both"])

DB_CONFIG = {
    "host":     os.getenv("PG_HOST", "localhost"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "dbname":   os.getenv("PG_DBNAME", "oliveyoung"),
    "user":     os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", ""),
}

def get_conn():
    return psycopg2.connect(**DB_CONFIG)

_embeddings: HuggingFaceEmbeddings | None = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings

def get_query_embedding(query: str) -> list[float]:
    return get_embeddings().embed_query(query)

_SKIN_WEIGHTS_CACHE: dict | None = None
_REVIEW_SUMMARY_CACHE: dict | None = None
_PRODUCT_STATS_CACHE:  dict | None = None


def load_product_stats() -> dict:
    """product_stats.json 로드 후 캐싱"""
    global _PRODUCT_STATS_CACHE
    if _PRODUCT_STATS_CACHE is None:
        if STATS_PATH.exists():
            _PRODUCT_STATS_CACHE = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        else:
            _PRODUCT_STATS_CACHE = {}
    return _PRODUCT_STATS_CACHE  # goods_no → metadata dict

_SKIN_WEIGHTS_DEFAULT = {
    "지성":    {"non_sticky_pos": 0.35, "fresh_pos": 0.30, "mild_pos": 0.20, "moisture_pos": 0.15},
    "건성":    {"moisture_pos": 0.40, "mild_pos": 0.25, "fresh_pos": 0.20, "non_sticky_pos": 0.15},
    "복합성":  {"mild_pos": 0.30, "fresh_pos": 0.25, "moisture_pos": 0.25, "non_sticky_pos": 0.20},
    "민감성":  {"mild_pos": 0.45, "no_whitecast_pos": 0.25, "moisture_pos": 0.20, "fresh_pos": 0.10},
    "중성":    {"mild_pos": 0.30, "fresh_pos": 0.25, "moisture_pos": 0.25, "non_sticky_pos": 0.20},
    "복합건성": {"moisture_pos": 0.35, "mild_pos": 0.30, "fresh_pos": 0.20, "non_sticky_pos": 0.15},
    "복합지성": {"non_sticky_pos": 0.35, "fresh_pos": 0.30, "mild_pos": 0.20, "moisture_pos": 0.15},
}

def load_skin_type_weights() -> dict:
    global _SKIN_WEIGHTS_CACHE
    if _SKIN_WEIGHTS_CACHE is None:
        if WEIGHTS_PATH.exists():
            with open(WEIGHTS_PATH, encoding="utf-8") as f:
                _SKIN_WEIGHTS_CACHE = json.load(f)
        else:
            _SKIN_WEIGHTS_CACHE = _SKIN_WEIGHTS_DEFAULT
    return _SKIN_WEIGHTS_CACHE


def load_review_summary() -> dict:
    """
    review_summary.json 로드 후 캐싱
    반환: {goods_no: {"goods_name": str, "metadata": {...}}}
    → goods_name 포함해서 _format_insights에서 제품명 표시 가능
    """
    global _REVIEW_SUMMARY_CACHE
    if _REVIEW_SUMMARY_CACHE is None:
        if REVIEW_SUMMARY_PATH.exists():
            with open(REVIEW_SUMMARY_PATH, encoding="utf-8") as f:
                raw = json.load(f)
            _REVIEW_SUMMARY_CACHE = {
                item["goods_no"]: {
                    "goods_name": item.get("goods_name", ""),
                    "metadata":   item.get("metadata", {}),
                }
                for item in raw
                if item.get("goods_no")
            }
        else:
            _REVIEW_SUMMARY_CACHE = {}
    return _REVIEW_SUMMARY_CACHE

def calculate_weights(user_needs: list[str], skin_type: str, defaults: dict) -> dict:
    skin_defaults = defaults.get(skin_type, defaults.get("복합성", {}))
    if not user_needs:
        return skin_defaults
    # 사용자가 직접 말한 니즈 0.85 : 피부타입 기본 가중치 0.15
    # (0.4:0.6일 때는 8개 속성에 고르게 분산된 기본 가중치 때문에
    #  모든 속성에서 상위권인 '올라운더' 제품이 니즈와 무관하게 1위를 독식)
    base     = {k: v * 0.15 for k, v in skin_defaults.items()}
    explicit = 0.85 / len(user_needs)
    for need in user_needs:
        col = NEEDS_TO_STAT.get(need)
        if col:
            base[col] = base.get(col, 0) + explicit
    total = sum(base.values())
    return {k: round(v / total, 4) for k, v in base.items()}

def rows_to_docs(rows: list[tuple], columns: list[str]) -> list[Document]:
    docs = []
    for row in rows:
        data    = dict(zip(columns, row))
        content = data.pop("content", "")
        docs.append(Document(page_content=content, metadata=data))
    return docs

def resolve_goods_no(product_name: str) -> str:
    results = resolve_goods_nos(product_name, limit=1)
    return results[0] if results else ""

def resolve_goods_nos(product_name: str, limit: int = 3) -> list[str]:
    """
    제품명 → goods_no 목록 반환 (최대 limit개)

    매칭 전략 (순서대로 시도):
    1. goods_no 형식(A + 14자리) → 그대로 반환
    2. 전체 문자열 ILIKE 매칭
    3. 핵심 키워드 분리 AND 매칭
       예) "풀리 선크림" → ILIKE '%풀리%' → "풀리 쌀 세라 수분 선크림" 매칭
    4. 첫 번째 키워드만으로 fallback
    """
    if product_name.startswith("A") and len(product_name) == 15:
        return [product_name]
    try:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                # 1차: 전체 문자열 매칭
                cur.execute(
                    "SELECT goods_no FROM products WHERE goods_name ILIKE %s LIMIT %s",
                    (f"%{product_name}%", limit)
                )
                rows = cur.fetchall()
                if rows:
                    return [row[0] for row in rows]

                # 2차: 핵심 키워드 분리 AND 매칭
                # 불용어 제거 후 브랜드명·특징어만 남겨서 매칭
                stopwords = {"선크림", "선세럼", "선크", "선", "수분", "크림", "세럼", "스틱", "해줘", "추천"}
                keywords = [
                    w for w in product_name.split()
                    if w not in stopwords and len(w) >= 2
                ]
                if not keywords:
                    return []

                # 각 키워드를 AND 조건으로 연결
                where_parts = " AND ".join(
                    "goods_name ILIKE %s" for _ in keywords
                )
                params = [f"%{kw}%" for kw in keywords] + [limit]
                cur.execute(
                    f"SELECT goods_no FROM products WHERE {where_parts} LIMIT %s",
                    params
                )
                rows = cur.fetchall()
                if rows:
                    return [row[0] for row in rows]

                # 3차: 첫 번째 키워드만으로 fallback
                cur.execute(
                    "SELECT goods_no FROM products WHERE goods_name ILIKE %s LIMIT %s",
                    (f"%{keywords[0]}%", limit)
                )
                return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()
    except Exception:
        return []

ASPECT_POS_COLS = [
    "fresh_pos", "non_sticky_pos", "moisture_pos", "mild_pos",
    "no_whitecast_pos", "no_pilling_pos", "longevity_pos", "no_eye_irritation_pos",
]
# 베이지안 평활화 강도: 언급 수가 이 값보다 적으면 긍정률이 평균 쪽으로 크게 당겨짐
SMOOTHING_K = 20


def _aspect_scores(rows: list[dict], weights: dict) -> dict[str, float]:
    """
    제품별 aspect 점수 = Σ weight × z(평활화된 긍정률)

    이전 방식(긍정률 × 언급수/리뷰수 × 가중치)의 문제:
      리뷰 하나에 속성을 많이 언급하는 제품(풀리: 리뷰당 4.75개, 중앙값 2.53개)이
      모든 속성 점수가 부풀려져 어떤 질문이든 1위가 됨
    현재 방식:
      - 언급 수는 '신뢰도'로만 사용 (베이지안 평활화) → 수다스러운 리뷰가 점수를 키우지 못함
      - 속성별 z-score → 모든 제품이 긍정률 ~1.0인 속성(산뜻함)이 순위를 지배하지 못함
    """
    import statistics
    smoothed: dict[str, dict[str, float]] = {}
    for col in ASPECT_POS_COLS:
        if weights.get(col, 0) <= 0:
            continue
        mention_col = col.replace("_pos", "_mention")
        vals = [(r[col], r.get(mention_col) or 0) for r in rows if r.get(col) is not None]
        if not vals:
            continue
        prior = sum(p * m for p, m in vals) / max(sum(m for _, m in vals), 1)
        per = {}
        for r in rows:
            p, m = r.get(col), r.get(mention_col) or 0
            per[r["goods_no"]] = prior if p is None else (p * m + prior * SMOOTHING_K) / (m + SMOOTHING_K)
        mu = statistics.mean(per.values())
        sd = statistics.pstdev(per.values()) or 1.0
        smoothed[col] = {g: (v - mu) / sd for g, v in per.items()}

    return {
        r["goods_no"]: sum(weights[col] * z[r["goods_no"]] for col, z in smoothed.items())
        for r in rows
    }


def _rank_norm(order: list[str]) -> dict[str, float]:
    """순위 → 0~1 점수 (1위 = 1.0)"""
    n = len(order) or 1
    return {g: (n - i) / n for i, g in enumerate(order)}


def search_by_product_stats(
    user_needs: list[str],
    skin_type: str,
    skin_tone_code: str = "",
    top_k: int = PRODUCT_K,
    already_recommended: list[str] | None = None,
) -> list[str]:
    """
    1단계: product_stats 스코어링 → 상위 제품 goods_no 반환

    1. aspect 점수 (피부타입 기본 가중치 0.15 + 사용자 니즈 0.85) → 상위 top_k × 3 후보
    2. 신뢰도 블렌딩: aspect 순위 0.9 + (평점·리뷰수) 0.1
       (질문과 무관한 고정값이라 동점 처리용으로만 — 0.2일 땐 리뷰 최다 제품이 순위를 뒤집음)
    3. 피부타입 보정: 해당 제품의 주 사용 피부타입이 요청과 같으면 +0.05
    4. 피부톤 블렌딩 (skin_tone_code 있을 때): 위 순위 0.6 + 피부톤 리뷰 비중 0.4
    5. 이미 추천한 제품 × 0.5 패널티 (다양성 확보)
    """
    import math

    weights = calculate_weights(user_needs, skin_type, load_skin_type_weights())

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM product_stats")
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        if not rows:
            return []

        # ── 1. aspect 점수 → 후보 풀 ──────────────────────────────────────────
        aspect = _aspect_scores(rows, weights)
        candidates = sorted(aspect, key=lambda g: aspect[g], reverse=True)[: top_k * 3]
        scores = _rank_norm(candidates)

        # ── 2. 신뢰도 (평점 정규화 × log 리뷰수), 동점 처리용 ────────────────
        summary = load_review_summary()
        metas = {g: (v.get("metadata", {}) if isinstance(v, dict) else {}) for g, v in summary.items()}
        ratings = [m["stat_rating"] for m in metas.values() if "stat_rating" in m]
        counts  = [m["stat_review_count"] for m in metas.values() if "stat_review_count" in m]
        if ratings and counts:
            r_min, r_max = min(ratings), max(ratings)
            r_range = (r_max - r_min) or 1.0
            log_max = math.log(max(counts) + 1)
            for g in candidates:
                m = metas.get(g, {})
                norm_rating = (m.get("stat_rating", 4.5) - r_min) / r_range
                norm_log    = math.log(m.get("stat_review_count", 0) + 1) / log_max
                trust = norm_rating * 0.5 + norm_log * 0.5
                scores[g] = scores[g] * 0.9 + trust * 0.1

        # ── 3. 피부타입 보정 ─────────────────────────────────────────────────
        skin_type_normalized = normalize_skin_type(skin_type) if skin_type else ""
        if skin_type_normalized:
            for g in candidates:
                if metas.get(g, {}).get("stat_skin_type_top") == skin_type_normalized:
                    scores[g] += 0.05

        # ── 4. 피부톤 블렌딩 ─────────────────────────────────────────────────
        if skin_tone_code:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT goods_no, COUNT(*) as cnt
                    FROM reviews
                    WHERE skin_tone = %s AND goods_no = ANY(%s)
                    GROUP BY goods_no
                """, (skin_tone_code, candidates))
                tone_counts = {row[0]: row[1] for row in cur.fetchall()}
            max_tone = max(tone_counts.values()) if tone_counts else 1
            type_rank = _rank_norm(sorted(candidates, key=lambda g: scores[g], reverse=True))
            scores = {
                g: type_rank[g] * 0.6 + (tone_counts.get(g, 0) / max_tone) * 0.4
                for g in candidates
            }

        # ── 5. 이미 추천한 제품 패널티 ───────────────────────────────────────
        if already_recommended:
            for g in candidates:
                if g in already_recommended:
                    scores[g] *= 0.5

        return sorted(candidates, key=lambda g: scores[g], reverse=True)[:top_k]
    finally:
        conn.close()

def search_balanced_by_products(
    query: str,
    goods_nos: list[str],
    skin_type_code: str = "",
    skin_tone_code: str = "",
    sample_k: int = SAMPLE_K,
) -> list[Document]:
    """
    2단계: 선정된 제품별 균등 샘플링
    skin_type + skin_tone 필터 + 유사도 정렬로 sample_k개씩 추출
    결과 없으면 skin_tone 제거 → skin_type만 → 전체 순으로 fallback
    """
    if not goods_nos:
        return []
    query_vec     = get_query_embedding(query)
    query_vec_str = "[" + ",".join(map(str, query_vec)) + "]"

    def _fetch(cur, gno, st, tone):
        if st and tone:
            cur.execute("""
                SELECT r.id, r.goods_no, p.goods_name, r.content, r.score,
                       r.is_repurchase, r.skin_type, r.skin_tone, r.created_at
                FROM reviews r JOIN products p ON r.goods_no = p.goods_no
                WHERE r.goods_no = %s AND r.skin_type = %s AND r.skin_tone = %s
                ORDER BY r.embedding <-> %s::vector LIMIT %s
            """, (gno, st, tone, query_vec_str, sample_k))
        elif tone:
            cur.execute("""
                SELECT r.id, r.goods_no, p.goods_name, r.content, r.score,
                       r.is_repurchase, r.skin_type, r.skin_tone, r.created_at
                FROM reviews r JOIN products p ON r.goods_no = p.goods_no
                WHERE r.goods_no = %s AND r.skin_tone = %s
                ORDER BY r.embedding <-> %s::vector LIMIT %s
            """, (gno, tone, query_vec_str, sample_k))
        elif st:
            cur.execute("""
                SELECT r.id, r.goods_no, p.goods_name, r.content, r.score,
                       r.is_repurchase, r.skin_type, r.skin_tone, r.created_at
                FROM reviews r JOIN products p ON r.goods_no = p.goods_no
                WHERE r.goods_no = %s AND r.skin_type = %s
                ORDER BY r.embedding <-> %s::vector LIMIT %s
            """, (gno, st, query_vec_str, sample_k))
        else:
            cur.execute("""
                SELECT r.id, r.goods_no, p.goods_name, r.content, r.score,
                       r.is_repurchase, r.skin_type, r.skin_tone, r.created_at
                FROM reviews r JOIN products p ON r.goods_no = p.goods_no
                WHERE r.goods_no = %s
                ORDER BY r.embedding <-> %s::vector LIMIT %s
            """, (gno, query_vec_str, sample_k))
        return cur.fetchall()

    all_docs = []
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            for goods_no in goods_nos:
                rows = _fetch(cur, goods_no, skin_type_code, skin_tone_code)
                cols = [d[0] for d in cur.description]
                # fallback 1: skin_tone 제거
                if not rows and skin_tone_code:
                    rows = _fetch(cur, goods_no, skin_type_code, "")
                    cols = [d[0] for d in cur.description]
                # fallback 2: 전체 필터 제거
                if not rows and (skin_type_code or skin_tone_code):
                    rows = _fetch(cur, goods_no, "", "")
                    cols = [d[0] for d in cur.description]
                all_docs.extend(rows_to_docs(rows, cols))
    finally:
        conn.close()
    return all_docs

def search_by_goods_no(query: str, goods_no: str, skin_type_code: str = "", k: int = RETRIEVE_K) -> list[Document]:
    query_vec     = get_query_embedding(query)
    query_vec_str = "[" + ",".join(map(str, query_vec)) + "]"
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if skin_type_code:
                cur.execute("""
                    SELECT r.id, r.goods_no, p.goods_name, r.content, r.score,
                           r.is_repurchase, r.skin_type, r.skin_tone, r.created_at
                    FROM reviews r JOIN products p ON r.goods_no = p.goods_no
                    WHERE r.goods_no = %s AND r.skin_type = %s
                    ORDER BY r.embedding <-> %s::vector LIMIT %s
                """, (goods_no, skin_type_code, query_vec_str, k))
            else:
                cur.execute("""
                    SELECT r.id, r.goods_no, p.goods_name, r.content, r.score,
                           r.is_repurchase, r.skin_type, r.skin_tone, r.created_at
                    FROM reviews r JOIN products p ON r.goods_no = p.goods_no
                    WHERE r.goods_no = %s
                    ORDER BY r.embedding <-> %s::vector LIMIT %s
                """, (goods_no, query_vec_str, k))
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            if not rows and skin_type_code:
                cur.execute("""
                    SELECT r.id, r.goods_no, p.goods_name, r.content, r.score,
                           r.is_repurchase, r.skin_type, r.skin_tone, r.created_at
                    FROM reviews r JOIN products p ON r.goods_no = p.goods_no
                    WHERE r.goods_no = %s
                    ORDER BY r.embedding <-> %s::vector LIMIT %s
                """, (goods_no, query_vec_str, k))
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
            return rows_to_docs(rows, cols)
    finally:
        conn.close()

def search_by_skin_type(query: str, skin_type_code: str, k: int = RETRIEVE_K) -> list[Document]:
    if not skin_type_code:
        return []
    query_vec     = get_query_embedding(query)
    query_vec_str = "[" + ",".join(map(str, query_vec)) + "]"
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT r.id, r.goods_no, p.goods_name, r.content, r.score,
                       r.is_repurchase, r.skin_type, r.skin_tone, r.created_at
                FROM reviews r JOIN products p ON r.goods_no = p.goods_no
                WHERE r.skin_type = %s
                ORDER BY r.embedding <-> %s::vector LIMIT %s
            """, (skin_type_code, query_vec_str, k))
            cols = [d[0] for d in cur.description]
            return rows_to_docs(cur.fetchall(), cols)
    finally:
        conn.close()

def search_by_negative(query: str, goods_no: str = "", k: int = RETRIEVE_K) -> list[Document]:
    query_vec     = get_query_embedding(query)
    query_vec_str = "[" + ",".join(map(str, query_vec)) + "]"
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if goods_no:
                cur.execute("""
                    SELECT r.id, r.goods_no, p.goods_name, r.content, r.score,
                           r.is_repurchase, r.skin_type, r.skin_tone, r.created_at
                    FROM reviews r JOIN products p ON r.goods_no = p.goods_no
                    WHERE r.goods_no = %s AND r.score <= 3 AND r.is_repurchase = FALSE
                    ORDER BY r.embedding <-> %s::vector LIMIT %s
                """, (goods_no, query_vec_str, k))
            else:
                cur.execute("""
                    SELECT r.id, r.goods_no, p.goods_name, r.content, r.score,
                           r.is_repurchase, r.skin_type, r.skin_tone, r.created_at
                    FROM reviews r JOIN products p ON r.goods_no = p.goods_no
                    WHERE r.score <= 3 AND r.is_repurchase = FALSE
                    ORDER BY r.embedding <-> %s::vector LIMIT %s
                """, (query_vec_str, k))
            cols = [d[0] for d in cur.description]
            return rows_to_docs(cur.fetchall(), cols)
    finally:
        conn.close()

def search_by_skin_tone(
    query: str,
    skin_tone_code: str,
    skin_type_code: str = "",
    k: int = RETRIEVE_K,
    per_product: int = SAMPLE_K,
) -> list[Document]:
    """
    피부톤 기반 제품별 균등 샘플링

    피부타입 없고 피부톤만 있는 경우 전용:
    - 해당 피부톤 유저 리뷰를 제품별로 per_product개씩 수집
    - 결과를 제품별로 묶어서 균등하게 반환 (특정 제품 편중 방지)
    - skin_type_code 있으면 추가 필터 적용
    - 결과 없으면 피부톤 필터만 → 전체 fallback
    """
    if not skin_tone_code:
        return []

    query_vec     = get_query_embedding(query)
    query_vec_str = "[" + ",".join(map(str, query_vec)) + "]"

    # 피부톤 유저가 리뷰한 제품 목록 먼저 추출
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # 해당 피부톤 리뷰가 있는 제품들 추출 (리뷰 많은 순)
            if skin_type_code:
                cur.execute("""
                    SELECT goods_no, COUNT(*) as cnt
                    FROM reviews
                    WHERE skin_tone = %s AND skin_type = %s
                    GROUP BY goods_no
                    ORDER BY cnt DESC
                    LIMIT %s
                """, (skin_tone_code, skin_type_code, PRODUCT_K * 2))
            else:
                cur.execute("""
                    SELECT goods_no, COUNT(*) as cnt
                    FROM reviews
                    WHERE skin_tone = %s
                    GROUP BY goods_no
                    ORDER BY cnt DESC
                    LIMIT %s
                """, (skin_tone_code, PRODUCT_K * 2))
            tone_goods = [row[0] for row in cur.fetchall()]

        # 해당 피부톤 리뷰가 없으면 전체 fallback
        if not tone_goods:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT r.id, r.goods_no, p.goods_name, r.content, r.score,
                           r.is_repurchase, r.skin_type, r.skin_tone, r.created_at
                    FROM reviews r JOIN products p ON r.goods_no = p.goods_no
                    ORDER BY r.embedding <-> %s::vector LIMIT %s
                """, (query_vec_str, k))
                cols = [d[0] for d in cur.description]
                return rows_to_docs(cur.fetchall(), cols)

        # 제품별 균등 샘플링
        all_docs = []
        with conn.cursor() as cur:
            for goods_no in tone_goods:
                if skin_type_code:
                    cur.execute("""
                        SELECT r.id, r.goods_no, p.goods_name, r.content, r.score,
                               r.is_repurchase, r.skin_type, r.skin_tone, r.created_at
                        FROM reviews r JOIN products p ON r.goods_no = p.goods_no
                        WHERE r.goods_no = %s AND r.skin_tone = %s AND r.skin_type = %s
                        ORDER BY r.embedding <-> %s::vector LIMIT %s
                    """, (goods_no, skin_tone_code, skin_type_code, query_vec_str, per_product))
                else:
                    cur.execute("""
                        SELECT r.id, r.goods_no, p.goods_name, r.content, r.score,
                               r.is_repurchase, r.skin_type, r.skin_tone, r.created_at
                        FROM reviews r JOIN products p ON r.goods_no = p.goods_no
                        WHERE r.goods_no = %s AND r.skin_tone = %s
                        ORDER BY r.embedding <-> %s::vector LIMIT %s
                    """, (goods_no, skin_tone_code, query_vec_str, per_product))
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
                # 해당 제품에 피부톤 리뷰 없으면 전체 리뷰로 fallback
                if not rows:
                    cur.execute("""
                        SELECT r.id, r.goods_no, p.goods_name, r.content, r.score,
                               r.is_repurchase, r.skin_type, r.skin_tone, r.created_at
                        FROM reviews r JOIN products p ON r.goods_no = p.goods_no
                        WHERE r.goods_no = %s
                        ORDER BY r.embedding <-> %s::vector LIMIT %s
                    """, (goods_no, query_vec_str, per_product))
                    cols = [d[0] for d in cur.description]
                    rows = cur.fetchall()
                all_docs.extend(rows_to_docs(rows, cols))

        return all_docs
    finally:
        conn.close()

def general_search(query: str, k: int = RETRIEVE_K) -> list[Document]:
    query_vec     = get_query_embedding(query)
    query_vec_str = "[" + ",".join(map(str, query_vec)) + "]"
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT r.id, r.goods_no, p.goods_name, r.content, r.score,
                       r.is_repurchase, r.skin_type, r.skin_tone, r.created_at
                FROM reviews r JOIN products p ON r.goods_no = p.goods_no
                ORDER BY r.embedding <-> %s::vector LIMIT %s
            """, (query_vec_str, k))
            cols = [d[0] for d in cur.description]
            return rows_to_docs(cur.fetchall(), cols)
    finally:
        conn.close()

def search_by_aspect(
    query: str,
    goods_no: str,
    aspect: str,
    skin_type_code: str = "",
    sentiment: str = "positive",
    k: int = 5,
) -> list[Document]:
    if aspect not in _VALID_ASPECTS:
        raise ValueError(f"허용되지 않는 aspect 컬럼: {aspect!r}")
    if sentiment not in _VALID_SENTIMENTS:
        raise ValueError(f"허용되지 않는 sentiment: {sentiment!r}")

    query_vec     = get_query_embedding(query)
    query_vec_str = "[" + ",".join(map(str, query_vec)) + "]"

    def _execute(cur, with_skin: bool) -> list:
        skin_clause = "AND r.skin_type = %s" if with_skin else ""
        if sentiment == "both":
            cur.execute(
                f"""SELECT r.id, r.goods_no, p.goods_name, r.content, r.score,
                           r.is_repurchase, r.skin_type, r.skin_tone, r.created_at,
                           r.{aspect} AS aspect_label
                    FROM reviews r JOIN products p ON r.goods_no = p.goods_no
                    WHERE r.goods_no = %s {skin_clause}
                      AND r.{aspect} = ANY(%s)
                    ORDER BY r.embedding <-> %s::vector LIMIT %s""",
                ([goods_no, skin_type_code, ["positive","negative"], query_vec_str, k]
                 if with_skin else
                 [goods_no, ["positive","negative"], query_vec_str, k])
            )
        else:
            cur.execute(
                f"""SELECT r.id, r.goods_no, p.goods_name, r.content, r.score,
                           r.is_repurchase, r.skin_type, r.skin_tone, r.created_at,
                           r.{aspect} AS aspect_label
                    FROM reviews r JOIN products p ON r.goods_no = p.goods_no
                    WHERE r.goods_no = %s {skin_clause}
                      AND r.{aspect} = %s
                    ORDER BY r.embedding <-> %s::vector LIMIT %s""",
                ([goods_no, skin_type_code, sentiment, query_vec_str, k]
                 if with_skin else
                 [goods_no, sentiment, query_vec_str, k])
            )
        return cur.fetchall()

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            rows = _execute(cur, with_skin=bool(skin_type_code))
            if not rows and skin_type_code:
                rows = _execute(cur, with_skin=False)
            cols = [d[0] for d in cur.description]
            return rows_to_docs(rows, cols)
    finally:
        conn.close()