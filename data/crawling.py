"""
data/crawling.py
올리브영 선크림 리뷰 크롤링

전략:
    피부타입별 필터로 각 40개씩 (A01~A05) = 200개
    필터 없이 50개 (None 포함)             =  50개
    제품당 최대 250개 × 30개 = 7,500개

실행:
    python data/crawling.py

출력:
    data/oliveyoung_sunscreen_reviews.json
"""

import requests
import json
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

headers = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15",
    "Content-Type": "application/json",
    "Origin": "https://m.oliveyoung.co.kr",
    "Referer": "https://m.oliveyoung.co.kr/",
}

GOODS_LIST = [
    "A000000232725",  # 달바 핑크 톤업 선크림          9,498개
    "A000000219553",  # 구달 맑은 어성초 진정 수분      34,300개
    "A000000246780",  # 라운드랩 자작나무 수분 선크림   44,956개
    "A000000180162",  # 닥터지 그린 마일드 업 선        25,188개
    "A000000186166",  # 라운드랩 자작나무 수분 톤업     11,389개
    "A000000232053",  # 체이싱래빗 올어바웃글로우        7,746개
    "A000000218862",  # 달바 그린 톤업 선크림            1,362개
    "A000000240562",  # 풀리 쌀 세라 수분 선크림         3,265개
    "A000000215559",  # 스킨1004 센텔라 워터핏 선세럼    6,811개
    "A000000218955",  # 비플레인 녹두 쿨링 수분 선크림   3,127개
    "A000000151904",  # 식물나라 워터프루프 선크림       11,708개
    "A000000200647",  # 아떼 비건 릴리프 무기자차         5,008개
    "A000000247874",  # 셀퓨전씨 레이저 UV 썬스크린      7,477개
    "A000000184353",  # 넘버즈인 3번 파데스킵 톤업       14,443개
    "A000000202359",  # 메이크프렘 수딩 핑크 톤업         4,833개
    "A000000211309",  # 코스알엑스 울트라 라이트 선세럼   3,045개
    "A000000224657",  # 조선미녀 맑은쌀 선크림            1,883개
    "A000000224140",  # 차앤박 애프터레이 톤업            2,698개
    "A000000219094",  # 에스트라 더마UV365 비타C          1,634개
    "A000000219051",  # 비플레인 선뮤즈 톤업              1,905개
    "A000000248226",  # 아비브 글루타치온좀 CC 선크림     1,036개
    "A000000209917",  # 제로이드 데일리 선크림            1,348개
    "A000000232672",  # 메디힐 마데카소사이드 수분 선세럼  8,582개
    "A000000200614",  # 달바 퍼플 톤업 선크림             7,202개
    "A000000184352",  # 넘버즈인 1번 물막선크림            7,489개
    "A000000227070",  # 유이크 톤업 선크림                2,385개
    "A000000190745",  # 로벡틴 아쿠아 수딩 선크림         1,201개
    "A000000200673",  # 리얼베리어 세라 수분 장벽 선크림   1,846개
    "A000000202491",  # 바이오힐보 콜라겐 톤업 선크림     1,269개
    "A000000207901",  # 아누아 어성초 실키 모이스처        3,600개
]

SKIN_TYPES = ["A01", "A02", "A03", "A04", "A05"]
API_URL = "https://m.oliveyoung.co.kr/review/api/v2/reviews"


def fetch_reviews_by_skin(goods_no, skin_type, pages=4):
    reviews = []
    goods_name = ""

    for page in range(pages):
        payload = {
            "goodsNumber": goods_no,
            "page": page,
            "size": 10,
            "sortType": "USEFUL_SCORE_DESC",
            "reviewType": "ALL",
        }
        if skin_type:
            payload["skinType"] = skin_type

        try:
            res = requests.post(API_URL, headers=headers, json=payload, timeout=10)
            if res.status_code != 200:
                break

            data = res.json().get("data", [])
            if not data:
                break

            if not goods_name and data:
                goods_name = data[0].get("goodsDto", {}).get("goodsName", goods_no)

            for r in data:
                reviews.append({
                    "goods_number": goods_no,
                    "goods_name": goods_name,
                    "category": "선크림",
                    "option": r.get("goodsDto", {}).get("optionName", ""),
                    "content": r.get("content", ""),
                    "score": r.get("reviewScore", 0),
                    "is_repurchase": r.get("isRepurchase", False),
                    "skin_type": r.get("profileDto", {}).get("skinType", ""),
                    "skin_tone": r.get("profileDto", {}).get("skinTone", ""),
                    "created_at": r.get("createdDateTime", ""),
                    "source": "oliveyoung",
                })
        except Exception as e:
            print(f"    warning ({skin_type}, page={page}): {e}")
            break

        time.sleep(0.3)

    return reviews, goods_name


def deduplicate(reviews):
    seen = set()
    result = []
    for r in reviews:
        key = r["content"].strip()[:100]
        if key and key not in seen:
            seen.add(key)
            result.append(r)
    return result


def get_all_reviews(goods_no):
    all_reviews = []
    goods_name = ""

    for skin_type in SKIN_TYPES:
        reviews, name = fetch_reviews_by_skin(goods_no, skin_type, pages=4)
        if name and not goods_name:
            goods_name = name
        all_reviews.extend(reviews)
        print(f"    {skin_type}: {len(reviews)}개")

    reviews, name = fetch_reviews_by_skin(goods_no, None, pages=5)
    if name and not goods_name:
        goods_name = name
    all_reviews.extend(reviews)
    print(f"    None: {len(reviews)}개")

    deduped = deduplicate(all_reviews)
    return deduped, goods_name


if __name__ == "__main__":
    print(f"크롤링 시작: {len(GOODS_LIST)}개 제품\n")

    all_data = []
    for i, goods_no in enumerate(GOODS_LIST, 1):
        print(f"[{i}/{len(GOODS_LIST)}] {goods_no} 수집 중...")
        reviews, name = get_all_reviews(goods_no)
        all_data.extend(reviews)
        print(f"  → [{name[:30]}] 총 {len(reviews)}개\n")
        time.sleep(1)

    output = BASE_DIR / "oliveyoung_sunscreen_reviews.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 완료: 총 {len(all_data)}개 저장 → {output}")

    from collections import Counter
    skin_dist = Counter(r["skin_type"] for r in all_data)
    print("\n[피부타입 분포]")
    for k, v in sorted(skin_dist.items()):
        print(f"  {k or 'None'}: {v}개")