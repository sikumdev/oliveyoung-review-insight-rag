"""
data/crawl_review_summary.py
올리브영 상품별 리뷰 통계 API 크롤링

API 엔드포인트:
    GET https://m.oliveyoung.co.kr/review/api/v2/reviews/{goodsNo}/stats

실행:
    python data/crawl_review_summary.py

출력:
    data/review_summary.json
"""

import json
import time
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ── 크롤링 전제 조건 ──────────────────────────────────────────────────────────
# 제품당 리뷰 5개 미만인 제품은 GOODS_LIST에서 제외한다.
#
# 근거:
#   - cmp_subgraph에서 goods_no only fallback(피부타입 조건 제거) 후에도
#     충분한 리뷰가 확보되어야 LLM 요약 품질이 보장됨
#   - 리뷰 1~2개로 LLM 요약 시 품질 저하 → fallback 임계값(3개)보다
#     크롤링 최소 기준(5개)이 높아야 fallback 후에도 안전
#   - ChromaDB에 등록된 제품은 이 조건으로 리뷰 5개 이상을 보장
#     → goods_no only fallback 후 결과 0건 케이스를 파이프라인 단계에서 차단
#
# 코드 레벨에서도 no_reviews_found 빈 요약 안전망을 유지하여 이중 방어.

GOODS_LIST = [
    "A000000249451",  # 달바
    "A000000248619",  # 라운드랩
    "A000000232725",  # 구달
    "A000000219553",  # 토리든
    "A000000246780",  # 넘버즈인
    "A000000180162",  # 아로마티카
    "A000000186166",  # 에스트라
    "A000000232053",  # 닥터지
    "A000000218862",  # 에뛰드
    "A000000204631",  # 미쟝센
]

API_URL = "https://m.oliveyoung.co.kr/review/api/v2/reviews/{goods_no}/stats"
HEADERS = {
    "User-Agent":      "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15",
    "Referer":         "https://m.oliveyoung.co.kr/",
    "Accept":          "application/json",
    "Accept-Language": "ko-KR,ko;q=0.9",
}

# 피부타입 answerName → 메타데이터 키
SKIN_LABEL_MAP = {
    "복합성에 좋아요": "complex",
    "건성에 좋아요":   "dry",
    "지성에 좋아요":   "oily",
    "민감성에 좋아요": "sensitive",
    "중성에 좋아요":   "normal",
}

# questionName → 메타데이터 키 (첫 번째 항목 = 긍정 최고값)
QUESTION_KEY_MAP = {
    "발림성": "spreadability_score",   # 아주 만족해요 비율
    "자극도": "mildness_score",        # 자극없이 순해요 비율
    "보습력": "moisture_score",        # 아주 촉촉해요 비율
    "지속력": "longevity_score",       # 오래 지속돼요 비율
    "백탁":   "no_whitecast_score",    # 백탁 없어요 비율
    "끈적임": "non_sticky_score",      # 끈적임 없어요 비율
}


def fetch_stats(goods_no: str) -> dict:
    url  = API_URL.format(goods_no=goods_no)
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


def flatten_for_metadata(raw: dict) -> dict:
    """
    API 응답 → ChromaDB 메타데이터 형태 변환

    출력 예시:
    {
        "stat_rating":             4.9,
        "stat_review_count":       53,
        "stat_spreadability":      0.91,
        "stat_mildness":           0.75,
        "stat_skin_type_top":      "복합성",
        "stat_skin_complex_pct":   0.43,
        "stat_skin_dry_pct":       0.34,
        "stat_skin_oily_pct":      0.23,
    }

    stat_ prefix: 올리브영 공식 통계 데이터임을 명시 (리뷰 텍스트 기반과 구분)
    """
    data = raw.get("data", {})
    meta = {}

    # ── 평점 / 리뷰 수 ──────────────────────────────────────────────────────
    rating_dist = data.get("ratingDistribution", {})
    meta["stat_rating"]       = rating_dist.get("averageRating")
    meta["stat_review_count"] = data.get("reviewCount")

    # ── satisfactionStats 파싱 ───────────────────────────────────────────────
    for stat in data.get("satisfactionStats", []):
        q_name   = stat.get("questionName", "")
        answers  = stat.get("answerDtos", [])

        if q_name == "피부타입":
            # 피부타입별 비율 + top 추출
            top_pct  = -1
            top_skin = None
            for ans in answers:
                label = ans.get("answerName", "")
                pct   = ans.get("answerPercentage", 0) / 100
                key   = SKIN_LABEL_MAP.get(label)
                if key:
                    meta[f"stat_skin_{key}_pct"] = round(pct, 4)
                if pct > top_pct:
                    top_pct  = pct
                    top_skin = label.replace("에 좋아요", "")
            if top_skin:
                meta["stat_skin_type_top"] = top_skin

        elif q_name in QUESTION_KEY_MAP:
            # 첫 번째 항목(긍정 최고값) 비율 사용
            if answers:
                pct = answers[0].get("answerPercentage", 0) / 100
                meta[QUESTION_KEY_MAP[q_name]] = round(pct, 4)

    return meta


def crawl_all() -> list[dict]:
    results = []

    for goods_no in GOODS_LIST:
        print(f"[{goods_no}] 크롤링 중...")
        try:
            raw  = fetch_stats(goods_no)
            meta = flatten_for_metadata(raw)
            data = raw.get("data", {})

            # 리뷰 5개 미만 제품 제외 (크롤링 전제 조건)
            # cmp_subgraph goods_no only fallback 후에도 충분한 리뷰 확보 보장
            review_count = meta.get("stat_review_count") or 0
            if review_count < 5:
                print(f"  ⚠️  리뷰 {review_count}개 < 5 → 제외 (최소 기준 미달)")
                results.append({"goods_no": goods_no, "error": f"review_count={review_count} < 5", "metadata": {}})
                continue

            entry = {
                "goods_no":   goods_no,
                "goods_name": data.get("goodsName", ""),
                "metadata":   meta,
                "error":      None,
            }
            results.append(entry)

            print(f"  ✅ 평점: {meta.get('stat_rating')} | 리뷰: {meta.get('stat_review_count')}건")
            print(f"     발림성: {meta.get('spreadability_score')} | 자극도: {meta.get('mildness_score')} | 피부타입top: {meta.get('stat_skin_type_top')}")

        except Exception as e:
            print(f"  ❌ 실패: {e}")
            results.append({"goods_no": goods_no, "error": str(e), "metadata": {}})

        time.sleep(0.5)

    return results


if __name__ == "__main__":
    print(f"크롤링 시작: {len(GOODS_LIST)}개 상품\n")
    results = crawl_all()

    output = BASE_DIR / "review_summary.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    success = sum(1 for r in results if not r.get("error"))
    print(f"\n✅ 완료: {success}/{len(GOODS_LIST)}개 성공 → {output}")
    print("\n[메타데이터 샘플]")
    for r in results[:2]:
        if not r.get("error"):
            print(f"\n{r['goods_name'][:30]}...")
            print(json.dumps(r["metadata"], ensure_ascii=False, indent=2))