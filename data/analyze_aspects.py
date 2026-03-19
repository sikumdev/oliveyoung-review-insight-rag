"""
data/analyze_aspects.py
LLM으로 리뷰별 aspect 분류 + product_stats 집계

실행:
    python data/analyze_aspects.py

출력:
    data/review_aspects.json     리뷰별 aspect 분류 결과
    data/product_stats.json      제품별 통계 집계
    data/skin_type_weights.json  피부타입별 기본 가중치
"""

import json
import time
from pathlib import Path
from pydantic import BaseModel
from typing import Literal
from openai import OpenAI
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
REVIEWS_PATH   = BASE_DIR / "oliveyoung_sunscreen_reviews.json"
ASPECTS_PATH   = BASE_DIR / "review_aspects.json"
STATS_PATH     = BASE_DIR / "product_stats.json"
WEIGHTS_PATH   = BASE_DIR / "skin_type_weights.json"

client = OpenAI()

class ReviewAspectResult(BaseModel):
    fresh:             Literal["positive", "negative", "none"]
    non_sticky:        Literal["positive", "negative", "none"]
    moisture:          Literal["positive", "negative", "none"]
    mild:              Literal["positive", "negative", "none"]
    no_whitecast:      Literal["positive", "negative", "none"]
    no_pilling:        Literal["positive", "negative", "none"]
    longevity:         Literal["positive", "negative", "none"]
    no_eye_irritation: Literal["positive", "negative", "none"]

ASPECTS = list(ReviewAspectResult.model_fields.keys())

ASPECT_SYSTEM = """당신은 뷰티 리뷰 분석 전문가입니다.
선크림 리뷰를 읽고 아래 8개 속성에 대해 분류하세요.

분류 기준:
- positive: 해당 속성이 좋다는 의미가 담긴 경우 (직접/간접 모두 포함)
- negative: 해당 속성이 나쁘다는 의미가 담긴 경우 (직접/간접 모두 포함)
- none: 언급 자체가 없거나 판단 불가한 경우

positive 예시:
- "촉촉해요" → moisture: positive
- "촉촉해서 좋아요" → moisture: positive
- "끈적임 없어서 만족해요" → non_sticky: positive
- "산뜻하게 발려요" → fresh: positive

negative 예시:
- "끈적거려요" → non_sticky: negative
- "좀 건조해요" → moisture: negative
- "밀림이 있어요" → no_pilling: negative
- "눈이 따가워요" → no_eye_irritation: negative

none 예시:
- "가격 대비 좋아요" → 모든 속성: none
- "재구매했어요" → 모든 속성: none
- "약간 촉촉한 것 같기도 해요" → moisture: none (너무 애매)

속성 정의:
- fresh: 산뜻함/가벼움/보송함
- non_sticky: 끈적임 없음
- moisture: 보습력/촉촉함
- mild: 자극없음/순함/트러블없음
- no_whitecast: 백탁없음
- no_pilling: 밀림없음/화잘먹
- longevity: 지속력/오래감
- no_eye_irritation: 눈시림없음

"""


def classify_review(content: str) -> dict | None:
    try:
        msg = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            max_tokens=200,
            messages=[
                {"role": "system", "content": ASPECT_SYSTEM},
                {"role": "user", "content": f"리뷰: {content[:500]}"},
            ],
            response_format=ReviewAspectResult,
        )
        return msg.choices[0].message.parsed.model_dump()
    except Exception as e:
        print(f"    분류 실패: {e}")
        return None


def build_product_stats(reviews, aspects_map):
    stats = defaultdict(lambda: {
        "total_reviews": 0,
        **{f"{a}_pos": 0 for a in ASPECTS},
        **{f"{a}_mention": 0 for a in ASPECTS},
    })

    for r in reviews:
        key = r["goods_number"]
        review_id = f"{r['goods_number']}_{r['created_at']}_{r['content'][:30]}"
        result = aspects_map.get(review_id)
        if not result:
            continue

        stats[key]["total_reviews"] += 1
        for aspect in ASPECTS:
            val = result.get(aspect, "none")
            if val in ("positive", "negative"):
                stats[key][f"{aspect}_mention"] += 1
                if val == "positive":
                    stats[key][f"{aspect}_pos"] += 1

    final = {}
    for goods_no, s in stats.items():
        entry = {"goods_no": goods_no, "total_reviews": s["total_reviews"]}
        for aspect in ASPECTS:
            mention = s[f"{aspect}_mention"]
            pos = s[f"{aspect}_pos"]
            entry[f"{aspect}_mention"] = mention
            entry[f"{aspect}_pos"] = round(pos / mention, 4) if mention >= 3 else None
        final[goods_no] = entry

    return final


def build_skin_type_weights(reviews, aspects_map):
    SKIN_MAP = {
        "A01": "지성", "A02": "건성", "A03": "복합성",
        "A04": "민감성", "A05": "중성",
        "A06": "복합건성", "A07": "복합지성",
    }

    skin_counts = defaultdict(lambda: defaultdict(int))

    for r in reviews:
        skin = r.get("skin_type", "")
        if not skin or skin not in SKIN_MAP:
            continue
        review_id = f"{r['goods_number']}_{r['created_at']}_{r['content'][:30]}"
        result = aspects_map.get(review_id)
        if not result:
            continue
        skin_kr = SKIN_MAP[skin]
        for aspect in ASPECTS:
            if result.get(aspect) in ("positive", "negative"):
                skin_counts[skin_kr][aspect] += 1

    weights = {}
    for skin_kr, counts in skin_counts.items():
        total = sum(counts.values())
        if total == 0:
            continue
        raw = {a: max(counts.get(a, 0) / total, 0.01) for a in ASPECTS}
        total2 = sum(raw.values())
        weights[skin_kr] = {f"{a}_pos": round(v / total2, 4) for a, v in raw.items()}

    return weights


def main():
    with open(REVIEWS_PATH, encoding="utf-8") as f:
        reviews = json.load(f)
    print(f"총 리뷰 수: {len(reviews)}개\n")

    if ASPECTS_PATH.exists():
        with open(ASPECTS_PATH, encoding="utf-8") as f:
            aspects_map = json.load(f)
        print(f"체크포인트 로드: {len(aspects_map)}개 기분류 완료\n")
    else:
        aspects_map = {}

    new_count = 0
    for i, r in enumerate(reviews):
        review_id = f"{r['goods_number']}_{r['created_at']}_{r['content'][:30]}"

        if review_id in aspects_map:
            continue

        if len(r["content"].strip()) < 5:
            aspects_map[review_id] = {a: "none" for a in ASPECTS}
            continue

        result = classify_review(r["content"])
        if result:
            aspects_map[review_id] = result
            new_count += 1

        if new_count % 50 == 0 and new_count > 0:
            with open(ASPECTS_PATH, "w", encoding="utf-8") as f:
                json.dump(aspects_map, f, ensure_ascii=False, indent=2)
            print(f"  체크포인트 저장: {len(aspects_map)}개 완료")

        if i % 100 == 0:
            print(f"  진행: {i}/{len(reviews)} ({i/len(reviews)*100:.1f}%)")

        time.sleep(0.1)

    with open(ASPECTS_PATH, "w", encoding="utf-8") as f:
        json.dump(aspects_map, f, ensure_ascii=False, indent=2)
    print(f"\n분류 완료: {len(aspects_map)}개 → {ASPECTS_PATH}")

    print("\nproduct_stats 집계 중...")
    stats = build_product_stats(reviews, aspects_map)
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"product_stats 저장 → {STATS_PATH}")

    print("\nskin_type_weights 계산 중...")
    weights = build_skin_type_weights(reviews, aspects_map)
    with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)
    print(f"skin_type_weights 저장 → {WEIGHTS_PATH}")

    print("\n[product_stats 샘플]")
    for goods_no, s in list(stats.items())[:2]:
        print(f"\n{goods_no}:")
        print(f"  total: {s['total_reviews']}개")
        for a in ASPECTS:
            pos = s.get(f"{a}_pos")
            mention = s.get(f"{a}_mention", 0)
            if mention >= 3:
                print(f"  {a}: {pos} ({mention}건 언급)")

    print("\n[skin_type_weights 샘플]")
    for skin, w in list(weights.items())[:2]:
        print(f"\n{skin}: {w}")


if __name__ == "__main__":
    main()