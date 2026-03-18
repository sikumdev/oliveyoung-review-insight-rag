"""
data/preprocess.py
원본 크롤링 데이터 → ChromaDB 투입용 processed_reviews.json 생성

변경사항:
    review_summary.json (올리브영 공식 통계) 과 join하여
    상품별 stat_* 메타데이터 추가

실행:
    python data/preprocess.py
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

sys.path.append(str(Path(__file__).resolve().parents[1]))
from rag.retriever import SKIN_TYPE_MAP, SKIN_TONE_MAP

CODE_TO_SKIN_TYPE = {v: k for k, v in SKIN_TYPE_MAP.items() if len(k) >= 2}
CODE_TO_SKIN_TONE = {v: k for k, v in SKIN_TONE_MAP.items() if len(k) >= 2}


def clean_goods_name(name: str) -> str:
    name = re.sub(r'\[.*?\]', '', name).strip()
    name = re.sub(r'\s*\(.*?\)', '', name).strip()
    name = re.sub(r'\s*\d+(?:\.\d+)?(?:ml|g|L)\s*', ' ', name).strip()
    name = re.sub(r'\s*(기획|듀오|단품|\+\d+|\d+\+\d+)$', '', name).strip()
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def clean_content(text: str) -> str:
    if not text:
        return ""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def load_summary_map(summary_path: str) -> dict[str, dict]:
    """
    review_summary.json → goods_no 기준 딕셔너리
    {
        "A000000249451": {
            "stat_rating": 4.9,
            "stat_review_count": 53,
            "stat_spreadability": 0.91,
            ...
        }
    }
    없으면 빈 dict 반환 (review_summary.json이 없어도 동작)
    """
    path = Path(summary_path)
    if not path.exists():
        print(f"⚠️  review_summary.json 없음 → stat_* 메타데이터 미포함")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        summaries = json.load(f)

    summary_map = {}
    for s in summaries:
        if not s.get("error"):
            summary_map[s["goods_no"]] = s.get("metadata", {})

    print(f"✅ review_summary.json 로드: {len(summary_map)}개 상품")
    return summary_map


def preprocess(input_path: str, output_path: str, summary_path: str = "") -> list[dict]:
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    print(f"원본 리뷰 수: {len(raw)}")

    # 올리브영 공식 통계 로드
    summary_map = load_summary_map(
        summary_path or str(Path(input_path).parent / "review_summary.json")
    )

    processed = []
    skipped   = 0

    for i, r in enumerate(raw):
        content = clean_content(r.get("content", ""))

        if len(content) < 10:
            skipped += 1
            continue

        skin_type_code = r.get("skin_type") or ""
        skin_tone_code = r.get("skin_tone") or ""
        score          = r.get("score", 0)
        goods_no       = r.get("goods_number", "")

        # 올리브영 공식 통계 메타데이터 join
        stat_meta = summary_map.get(goods_no, {})

        processed.append({
            "document": content,
            "metadata": {
                # ── 기본 리뷰 메타 ───────────────────────────────────────
                "id":               f"review_{i:04d}",
                "goods_number":     goods_no,
                "goods_name":       clean_goods_name(r.get("goods_name", "")),
                "goods_name_raw":   r.get("goods_name", ""),
                "category":         r.get("category", "선크림"),
                "option":           r.get("option", ""),
                "score":            score,
                "score_normalized": round(score / 5.0, 2),
                "is_repurchase":    bool(r.get("is_repurchase", False)),
                "skin_type":        skin_type_code,
                "skin_type_label":  CODE_TO_SKIN_TYPE.get(skin_type_code, ""),
                "skin_tone":        skin_tone_code,
                "skin_tone_label":  CODE_TO_SKIN_TONE.get(skin_tone_code, ""),
                "created_at":       r.get("created_at", ""),
                "source":           r.get("source", "oliveyoung"),

                # ── 올리브영 공식 통계 (stat_ prefix) ───────────────────
                # 전체 리뷰 집계 기반 → 리뷰 텍스트 분석과 구분
                "stat_rating":           stat_meta.get("stat_rating"),
                "stat_review_count":     stat_meta.get("stat_review_count"),
                "stat_spreadability":    stat_meta.get("spreadability_score"),   # 발림성
                "stat_mildness":         stat_meta.get("mildness_score"),         # 자극도
                "stat_moisture":         stat_meta.get("moisture_score"),         # 보습력
                "stat_longevity":        stat_meta.get("longevity_score"),        # 지속력
                "stat_no_whitecast":     stat_meta.get("no_whitecast_score"),     # 백탁없음
                "stat_non_sticky":       stat_meta.get("non_sticky_score"),       # 끈적임없음
                "stat_skin_type_top":    stat_meta.get("stat_skin_type_top"),     # 가장 많은 피부타입
                "stat_skin_complex_pct": stat_meta.get("stat_skin_complex_pct"), # 복합성 비율
                "stat_skin_dry_pct":     stat_meta.get("stat_skin_dry_pct"),     # 건성 비율
                "stat_skin_oily_pct":    stat_meta.get("stat_skin_oily_pct"),    # 지성 비율
                "stat_skin_sensitive_pct": stat_meta.get("stat_skin_sensitive_pct"),
            }
        })

    print(f"필터링 제거: {skipped}개 (10자 미만)")
    print(f"최종 리뷰 수: {len(processed)}")

    print("\n상품별 리뷰 수:")
    for name, cnt in Counter(r["metadata"]["goods_name"] for r in processed).most_common():
        print(f"  {name}: {cnt}개")

    # stat 데이터 포함 여부 확인
    stat_count = sum(1 for r in processed if r["metadata"].get("stat_rating") is not None)
    print(f"\nstat_* 메타 포함 리뷰: {stat_count}개 / {len(processed)}개")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 저장 완료 → {output_path}")
    return processed


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    preprocess(
        input_path=str(base / "oliveyoung_sunscreen_reviews.json"),
        output_path=str(base / "processed_reviews.json"),
    )