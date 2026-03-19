"""
data/migrate_to_pg.py
PostgreSQL 마이그레이션

순서:
    1. products 테이블 ← review_summary.json
    2. reviews 테이블  ← oliveyoung_sunscreen_reviews.json + 임베딩 생성
    3. product_stats   ← product_stats.json

실행:
    python data/migrate_to_pg.py

전제:
    - PostgreSQL oliveyoung DB 생성 완료
    - pgvector 확장 활성화 완료
    - analyze_aspects.py 실행 완료 (product_stats.json 존재)
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

REVIEW_SUMMARY_PATH = BASE_DIR / "review_summary.json"
REVIEWS_PATH        = BASE_DIR / "oliveyoung_sunscreen_reviews.json"
STATS_PATH          = BASE_DIR / "product_stats.json"

DB_CONFIG = {
    "host":     os.getenv("PG_HOST", "localhost"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "dbname":   os.getenv("PG_DBNAME", "oliveyoung"),
    "user":     os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", ""),
}

EMBED_MODEL = "jhgan/ko-sroberta-multitask"
BATCH_SIZE  = 64  # 임베딩 배치 크기
CHECKPOINT  = BASE_DIR / "migrate_checkpoint.json"  # 이어하기용


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


# ── 1. products 테이블 ────────────────────────────────────────────────────────

def migrate_products(conn):
    print("\n[1/3] products 테이블 마이그레이션...")

    with open(REVIEW_SUMMARY_PATH, encoding="utf-8") as f:
        summaries = json.load(f)

    rows = []
    for item in summaries:
        if item.get("error"):
            continue
        m = item.get("metadata", {})
        rows.append((
            item["goods_no"],
            item["goods_name"],
            m.get("stat_rating"),
            m.get("stat_review_count"),
            m.get("stat_skin_type_top"),
            m.get("stat_skin_oily_pct"),
            m.get("stat_skin_dry_pct"),
            m.get("stat_skin_complex_pct"),
            m.get("spreadability_score"),
            m.get("mildness_score"),
        ))

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO products (
                goods_no, goods_name, stat_rating, stat_review_count,
                stat_skin_type_top, stat_skin_oily_pct, stat_skin_dry_pct,
                stat_skin_complex_pct, spreadability_score, mildness_score
            ) VALUES %s
            ON CONFLICT (goods_no) DO UPDATE SET
                goods_name          = EXCLUDED.goods_name,
                stat_rating         = EXCLUDED.stat_rating,
                stat_review_count   = EXCLUDED.stat_review_count,
                stat_skin_type_top  = EXCLUDED.stat_skin_type_top,
                stat_skin_oily_pct  = EXCLUDED.stat_skin_oily_pct,
                stat_skin_dry_pct   = EXCLUDED.stat_skin_dry_pct,
                stat_skin_complex_pct = EXCLUDED.stat_skin_complex_pct,
                spreadability_score = EXCLUDED.spreadability_score,
                mildness_score      = EXCLUDED.mildness_score
        """, rows)
    conn.commit()
    print(f"  ✅ {len(rows)}개 제품 저장 완료")


# ── 2. reviews 테이블 (임베딩 포함) ──────────────────────────────────────────

def migrate_reviews(conn):
    print("\n[2/3] reviews 테이블 마이그레이션 (임베딩 포함)...")

    with open(REVIEWS_PATH, encoding="utf-8") as f:
        reviews = json.load(f)

    # 체크포인트 로드
    if CHECKPOINT.exists():
        with open(CHECKPOINT) as f:
            done_ids = set(json.load(f))
        print(f"  체크포인트 로드: {len(done_ids)}개 완료")
    else:
        done_ids = set()

    # 임베딩 모델 로드
    print(f"  임베딩 모델 로드 중: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL, model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
    print("  모델 로드 완료")

    # 미완료 리뷰만 처리
    pending = [
        r for r in reviews
        if f"{r['goods_number']}_{r['created_at']}_{r['content'][:30]}" not in done_ids
        and len(r["content"].strip()) >= 5
    ]
    print(f"  처리 대상: {len(pending)}개 (완료: {len(done_ids)}개)")

    # 배치 처리
    for i in range(0, len(pending), BATCH_SIZE):
        batch = pending[i:i + BATCH_SIZE]
        contents = [r["content"][:512] for r in batch]

        # 임베딩 생성
        emb_list = embeddings.embed_documents(contents)

        rows = []
        for r, emb in zip(batch, emb_list):
            rows.append((
                r["goods_number"],
                r["content"],
                r.get("score"),
                r.get("is_repurchase", False),
                r.get("skin_type") or None,
                r.get("skin_tone") or None,
                r.get("option", ""),
                r.get("created_at", ""),
                emb,
            ))

        with conn.cursor() as cur:
            execute_values(cur, """
                INSERT INTO reviews (
                    goods_no, content, score, is_repurchase,
                    skin_type, skin_tone, option_name, created_at, embedding
                ) VALUES %s
            """, rows, template="(%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)")
        conn.commit()

        # 체크포인트 저장
        for r in batch:
            done_ids.add(f"{r['goods_number']}_{r['created_at']}_{r['content'][:30]}")
        with open(CHECKPOINT, "w") as f:
            json.dump(list(done_ids), f)

        print(f"  진행: {min(i + BATCH_SIZE, len(pending))}/{len(pending)}")

    print(f"  ✅ 리뷰 임베딩 저장 완료")


# ── 3. product_stats 테이블 ───────────────────────────────────────────────────

def migrate_stats(conn):
    print("\n[3/3] product_stats 테이블 마이그레이션...")

    if not STATS_PATH.exists():
        print("  ⚠️  product_stats.json 없음 → analyze_aspects.py 먼저 실행하세요")
        return

    with open(STATS_PATH, encoding="utf-8") as f:
        stats = json.load(f)

    rows = []
    for goods_no, s in stats.items():
        rows.append((
            goods_no,
            s.get("total_reviews"),
            s.get("fresh_pos"),           s.get("fresh_mention"),
            s.get("non_sticky_pos"),      s.get("non_sticky_mention"),
            s.get("moisture_pos"),        s.get("moisture_mention"),
            s.get("mild_pos"),            s.get("mild_mention"),
            s.get("no_whitecast_pos"),    s.get("no_whitecast_mention"),
            s.get("no_pilling_pos"),      s.get("no_pilling_mention"),
            s.get("longevity_pos"),       s.get("longevity_mention"),
            s.get("no_eye_irritation_pos"), s.get("no_eye_irritation_mention"),
        ))

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO product_stats (
                goods_no, total_reviews,
                fresh_pos, fresh_mention,
                non_sticky_pos, non_sticky_mention,
                moisture_pos, moisture_mention,
                mild_pos, mild_mention,
                no_whitecast_pos, no_whitecast_mention,
                no_pilling_pos, no_pilling_mention,
                longevity_pos, longevity_mention,
                no_eye_irritation_pos, no_eye_irritation_mention
            ) VALUES %s
            ON CONFLICT (goods_no) DO UPDATE SET
                total_reviews             = EXCLUDED.total_reviews,
                fresh_pos                 = EXCLUDED.fresh_pos,
                fresh_mention             = EXCLUDED.fresh_mention,
                non_sticky_pos            = EXCLUDED.non_sticky_pos,
                non_sticky_mention        = EXCLUDED.non_sticky_mention,
                moisture_pos              = EXCLUDED.moisture_pos,
                moisture_mention          = EXCLUDED.moisture_mention,
                mild_pos                  = EXCLUDED.mild_pos,
                mild_mention              = EXCLUDED.mild_mention,
                no_whitecast_pos          = EXCLUDED.no_whitecast_pos,
                no_whitecast_mention      = EXCLUDED.no_whitecast_mention,
                no_pilling_pos            = EXCLUDED.no_pilling_pos,
                no_pilling_mention        = EXCLUDED.no_pilling_mention,
                longevity_pos             = EXCLUDED.longevity_pos,
                longevity_mention         = EXCLUDED.longevity_mention,
                no_eye_irritation_pos     = EXCLUDED.no_eye_irritation_pos,
                no_eye_irritation_mention = EXCLUDED.no_eye_irritation_mention,
                updated_at                = NOW()
        """, rows)
    conn.commit()
    print(f"  ✅ {len(rows)}개 제품 통계 저장 완료")


# ── pgvector 인덱스 재생성 ────────────────────────────────────────────────────

def rebuild_index(conn):
    print("\n[+] pgvector 인덱스 재생성...")
    with conn.cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS reviews_embedding_idx;")
        cur.execute("""
            CREATE INDEX reviews_embedding_idx
            ON reviews USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
    conn.commit()
    print("  ✅ 인덱스 재생성 완료")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    print("PostgreSQL 마이그레이션 시작")
    print(f"  DB: {DB_CONFIG['dbname']} @ {DB_CONFIG['host']}")

    conn = get_conn()
    try:
        if mode in ("all", "products"):
            migrate_products(conn)
        if mode in ("all", "reviews"):
            migrate_reviews(conn)
        if mode in ("all", "stats"):
            migrate_stats(conn)
            rebuild_index(conn)
        print("\n✅ 마이그레이션 완료")
    finally:
        conn.close()