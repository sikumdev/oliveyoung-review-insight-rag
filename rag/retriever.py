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
_REVIEW_SUMMARY_CACHE: dict | None = None  # goods_no → metadata dict

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
    base     = {k: v * 0.4 for k, v in skin_defaults.items()}
    explicit = 0.6 / len(user_needs)
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

def search_by_product_stats(
    user_needs: list[str],
    skin_type: str,
    skin_tone_code: str = "",
    top_k: int = PRODUCT_K,
    already_recommended: list[str] | None = None,
) -> list[str]:
    """
    1단계: product_stats SQL 스코어링 → 상위 제품 goods_no 반환

    스코어 = aspect 가중합 (피부타입 기반)
    skin_tone_code 있을 때:
      - 후보 풀을 top_k × 3개로 확장
      - 피부타입 스코어 0.6 + 피부톤 매칭 스코어 0.4 블렌딩
    already_recommended 있을 때:
      - 이미 추천한 제품 스코어 × 0.5 패널티 (다양성 확보)
      - 같은 세션에서 계속 같은 제품이 추천되는 것 방지
    """
    defaults   = load_skin_type_weights()
    weights    = calculate_weights(user_needs, skin_type, defaults)
    # aspect_pos × mention_ratio × weight
    # mention_ratio = mention / total_reviews
    # → "얼마나 좋냐" × "얼마나 언급됐냐" × "피부타입 가중치"
    # mention이 적은 aspect는 신뢰도를 낮춰서 과대평가 방지
    score_parts = []
    for col, w in weights.items():
        if col not in ["fresh_pos","non_sticky_pos","moisture_pos","mild_pos",
                       "no_whitecast_pos","no_pilling_pos","longevity_pos","no_eye_irritation_pos"]:
            continue
        mention_col = col.replace("_pos", "_mention")
        # mention_ratio = mention / total_reviews (0~1)
        # NULLIF로 total_reviews=0 방어
        score_parts.append(
            f"COALESCE({col}, 0.5) "
            f"* COALESCE({mention_col}::float / NULLIF(total_reviews, 0), 0.3) "
            f"* {w}"
        )
    score_expr = " + ".join(score_parts) or "0.5"

    # review_summary에서 신뢰도 점수 계산 (rating 정규화 × log 리뷰수)
    import math
    summary = load_review_summary()
    trust_scores: dict[str, float] = {}
    if summary:
        # summary 구조: {goods_no: {"goods_name": ..., "metadata": {...}}}
        metas = [v["metadata"] for v in summary.values() if isinstance(v, dict) and "metadata" in v]
        ratings = [m["stat_rating"] for m in metas if "stat_rating" in m]
        counts  = [m["stat_review_count"] for m in metas if "stat_review_count" in m]
        if not ratings or not counts:
            pass  # review_summary 없으면 trust_scores 비워두고 스킵
        else:
            r_min, r_max = min(ratings), max(ratings)
            r_range   = r_max - r_min if r_max != r_min else 1.0
            log_max   = math.log(max(counts) + 1)
            for gno, item in summary.items():
                meta = item.get("metadata", {}) if isinstance(item, dict) else {}
                rating    = meta.get("stat_rating", 4.5)
                rev_count = meta.get("stat_review_count", 0)
            norm_rating = (rating - r_min) / r_range          # 0~1 정규화
            norm_log    = math.log(rev_count + 1) / log_max   # 0~1 로그 정규화
            trust_scores[gno] = round(norm_rating * 0.5 + norm_log * 0.5, 4)

    sql = f"SELECT goods_no FROM product_stats ORDER BY ({score_expr}) DESC LIMIT %s"
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (top_k * 3,))  # 여유 있게 가져온 뒤 신뢰도로 재정렬
            candidates = [row[0] for row in cur.fetchall()]

        if trust_scores:
            # aspect 스코어 0.8 + 신뢰도 0.2 블렌딩
            # (SQL 스코어는 상대 순위만 사용 → rank 기반 정규화)
            ranked = {gno: (len(candidates) - i) / len(candidates)
                      for i, gno in enumerate(candidates)}
            final_scores = {
                gno: ranked.get(gno, 0) * 0.8 + trust_scores.get(gno, 0.5) * 0.2
                for gno in candidates
            }
            candidates = sorted(candidates, key=lambda g: final_scores.get(g, 0), reverse=True)

        # 3. 피부타입 분포 보정
        # skin_type이 있을 때 stat_skin_type_top과 맞지 않는 제품은 0.1 패널티
        skin_type_normalized = normalize_skin_type(skin_type) if skin_type else ""
        if skin_type_normalized and summary:
            final_scores_with_skin = {}
            for gno in candidates:
                base = final_scores.get(gno, ranked.get(gno, 0)) if trust_scores else ranked.get(gno, 0)
                meta = summary.get(gno, {})
                skin_top = meta.get("stat_skin_type_top", "")
                # 요청 피부타입과 주 사용 피부타입 일치 시 보너스
                if skin_top and skin_top == skin_type_normalized:
                    final_scores_with_skin[gno] = base + 0.05
                else:
                    final_scores_with_skin[gno] = base
            candidates = sorted(candidates, key=lambda g: final_scores_with_skin.get(g, 0), reverse=True)

        # ── 이미 추천한 제품 패널티 (다양성 확보) ───────────────────────────
        if already_recommended:
            penalized = {
                gno: (blended.get(gno, type_scores.get(gno, 0)) * 0.5
                      if gno in already_recommended
                      else blended.get(gno, type_scores.get(gno, 0)))
                for gno in candidates
            } if skin_tone_code else {
                gno: (final_scores.get(gno, ranked.get(gno, 0)) * 0.5
                      if gno in already_recommended
                      else final_scores.get(gno, ranked.get(gno, 0)))
                for gno in candidates
            }
            candidates = sorted(candidates, key=lambda g: penalized.get(g, 0), reverse=True)

        # ── 피부톤 블렌딩 ────────────────────────────────────────────────────
        # skin_tone_code 있을 때: 피부타입 스코어 0.6 + 피부톤 매칭 0.4
        # 피부톤 매칭 스코어 = 해당 제품의 피부톤 리뷰 수 / 전체 최대값
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
            # 현재 candidates 순위 기반 정규화 (피부타입 스코어 proxy)
            type_scores = {gno: (len(candidates) - i) / len(candidates)
                           for i, gno in enumerate(candidates)}
            blended = {
                gno: type_scores.get(gno, 0) * 0.6
                     + (tone_counts.get(gno, 0) / max_tone) * 0.4
                for gno in candidates
            }
            candidates = sorted(candidates, key=lambda g: blended.get(g, 0), reverse=True)

        return candidates[:top_k]
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