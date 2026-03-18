import requests
import json
import time

headers = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15",
    "Content-Type": "application/json",
    "Origin": "https://m.oliveyoung.co.kr",
    "Referer": "https://m.oliveyoung.co.kr/",
}

def get_reviews(goods_number, pages=10):
    all_reviews = []
    goods_name = ""

    for page in range(pages):
        payload = {
            "goodsNumber": goods_number,
            "page": page,
            "size": 10,
            "sortType": "USEFUL_SCORE_DESC",
            "reviewType": "ALL"
        }
        res = requests.post(
            "https://m.oliveyoung.co.kr/review/api/v2/reviews",
            headers=headers,
            json=payload
        )
        if res.status_code == 200:
            reviews = res.json().get("data", [])
            if not reviews:
                break
            # 첫 번째 리뷰에서 상품명 추출
            if not goods_name and reviews:
                goods_name = reviews[0].get("goodsDto", {}).get("goodsName", goods_number)

            for r in reviews:
                all_reviews.append({
                    "goods_number": goods_number,
                    "goods_name": goods_name,
                    "category": "선크림",
                    "option": r.get("goodsDto", {}).get("optionName", ""),
                    "content": r.get("content", ""),
                    "score": r.get("reviewScore", 0),
                    "is_repurchase": r.get("isRepurchase", False),
                    "skin_type": r.get("profileDto", {}).get("skinType", ""),
                    "skin_tone": r.get("profileDto", {}).get("skinTone", ""),
                    "created_at": r.get("createdDateTime", ""),
                    "source": "oliveyoung"
                })
        time.sleep(0.5)

    return all_reviews, goods_name

products = [
    "A000000249451", "A000000248619", "A000000232725",
    "A000000219553", "A000000246780", "A000000180162",
    "A000000186166", "A000000232053", "A000000218862",
    "A000000204631"
]

all_data = []
for goods_no in products:
    print(f"수집 중: {goods_no}")
    reviews, name = get_reviews(goods_no)
    all_data.extend(reviews)
    print(f"  → [{name}] {len(reviews)}개 완료")
    time.sleep(1)

with open("oliveyoung_sunscreen_reviews.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"\n✅ 총 {len(all_data)}개 저장 완료 → oliveyoung_sunscreen_reviews.json")