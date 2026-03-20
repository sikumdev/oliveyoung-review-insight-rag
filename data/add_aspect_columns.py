"""
data/add_aspect_columns.py

reviews 테이블에 aspect 컬럼 8개 추가 후
review_aspects.json 데이터를 UPDATE

실행:
    python data/add_aspect_columns.py

전제:
    - migrate_to_pg.py 실행 완료 (reviews 테이블 존재)
    - analyze_aspects.py 실행 완료 (review_aspects.json 존재)

추가되는 컬럼:
    fresh, non_sticky, moisture, mild,
    no_whitecast, no_pilling, longevity, no_eye_irritation
    값: 'positive' | 'negative' | 'none'
"""

import json
import os
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

load_dotenv()

BASE_DIR      = Path(__file__).resolve().parent
ASPECTS_PATH  = BASE_DIR / "review_aspects.json"

ASPECTS = [
    "fresh", "non_sticky", "moisture", "mild",
    "no_whitecast", "no_pilling", "longevity", "no_eye_irritation",
]

DB_CONFIG = {
    "host":     os.getenv("PG_HOST", "localhost"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "dbname":   os.getenv("PG_DBNAME", "oliveyoung"),
    "user":     os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASSWORD", ""),
}


def get_conn():
    return psycopg2.connect(**DB_CONFIG)


# ── 1단계: 컬럼 추가 ──────────────────────────────────────────────────────────

def add_columns(conn):
    print("[1/3] aspect 컬럼 추가 중...")
    with conn.cursor() as cur:
        for aspect in ASPECTS:
            # 이미 존재하면 무시
            cur.execute(f"""
                ALTER TABLE reviews
                ADD COLUMN IF NOT EXISTS {aspect}
                VARCHAR(10) DEFAULT 'none'
            """)
    conn.commit()
    print(f"  ✅ 컬럼 {len(ASPECTS)}개 추가 완료: {', '.join(ASPECTS)}")


# ── 2단계: review_aspects.json 로드 ──────────────────────────────────────────

def load_aspects_map() -> dict:
    print("\n[2/3] review_aspects.json 로드 중...")
    if not ASPECTS_PATH.exists():
        raise FileNotFoundError(
            f"{ASPECTS_PATH} 없음 → analyze_aspects.py 먼저 실행하세요"
        )
    with open(ASPECTS_PATH, encoding="utf-8") as f:
        aspects_map = json.load(f)
    print(f"  ✅ {len(aspects_map)}개 리뷰 레이블 로드")
    return aspects_map


# ── 3단계: DB reviews 행과 매칭해서 UPDATE ───────────────────────────────────

def update_aspects(conn, aspects_map: dict):
    """
    reviews 테이블의 각 행을 review_aspects.json의 key와 매칭해서 UPDATE

    key 형식: "{goods_no}_{created_at}_{content[:30]}"
    → reviews 테이블의 goods_no, created_at, content로 재조합해서 매칭
    """
    print("\n[3/3] aspect 레이블 UPDATE 중...")

    # 전체 reviews 읽기 (id, goods_no, created_at, content 앞 30자)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, goods_no, created_at, LEFT(content, 30)
            FROM reviews
            ORDER BY id
        """)
        rows = cur.fetchall()

    print(f"  전체 리뷰 수: {len(rows)}개")

    matched   = 0
    unmatched = 0
    batch     = []  # (fresh, non_sticky, ..., id)

    for row_id, goods_no, created_at, content_prefix in rows:
        key = f"{goods_no}_{created_at}_{content_prefix}"
        result = aspects_map.get(key)

        if result is None:
            unmatched += 1
            continue

        batch.append((
            result.get("fresh",             "none"),
            result.get("non_sticky",        "none"),
            result.get("moisture",          "none"),
            result.get("mild",              "none"),
            result.get("no_whitecast",      "none"),
            result.get("no_pilling",        "none"),
            result.get("longevity",         "none"),
            result.get("no_eye_irritation", "none"),
            row_id,
        ))
        matched += 1

        # 500개 단위로 배치 UPDATE
        if len(batch) >= 500:
            _batch_update(conn, batch)
            print(f"  진행: {matched}개 업데이트 완료...")
            batch = []

    # 나머지 처리
    if batch:
        _batch_update(conn, batch)

    conn.commit()
    print(f"\n  ✅ 매칭 성공: {matched}개")
    print(f"  ⚠️  매칭 실패: {unmatched}개 (key 불일치 — none 유지)")


def _batch_update(conn, batch: list):
    """배치 UPDATE — executemany보다 빠른 임시 테이블 방식"""
    with conn.cursor() as cur:
        execute_values(cur, """
            UPDATE reviews SET
                fresh             = data.fresh,
                non_sticky        = data.non_sticky,
                moisture          = data.moisture,
                mild              = data.mild,
                no_whitecast      = data.no_whitecast,
                no_pilling        = data.no_pilling,
                longevity         = data.longevity,
                no_eye_irritation = data.no_eye_irritation
            FROM (VALUES %s) AS data(
                fresh, non_sticky, moisture, mild,
                no_whitecast, no_pilling, longevity, no_eye_irritation,
                id
            )
            WHERE reviews.id = data.id::integer
        """, batch)
    conn.commit()


# ── 4단계: 검증 ───────────────────────────────────────────────────────────────

def verify(conn):
    print("\n[검증] aspect 컬럼 분포 확인...")
    with conn.cursor() as cur:
        for aspect in ASPECTS:
            cur.execute(f"""
                SELECT {aspect}, COUNT(*)
                FROM reviews
                GROUP BY {aspect}
                ORDER BY {aspect}
            """)
            counts = {row[0]: row[1] for row in cur.fetchall()}
            pos  = counts.get("positive", 0)
            neg  = counts.get("negative", 0)
            none = counts.get("none", 0)
            total = pos + neg + none
            print(f"  {aspect:20s}: positive={pos:4d}  negative={neg:4d}  none={none:5d}  (total={total})")


if __name__ == "__main__":
    print("=" * 55)
    print("reviews 테이블 aspect 컬럼 마이그레이션")
    print("=" * 55)

    conn = get_conn()
    try:
        aspects_map = load_aspects_map()
        add_columns(conn)
        update_aspects(conn, aspects_map)
        verify(conn)
        print("\n✅ 완료")
    finally:
        conn.close()