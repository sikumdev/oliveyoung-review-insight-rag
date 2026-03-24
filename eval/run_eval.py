"""
eval/run_eval.py

올리브영 선크림 RAG 챗봇 성능 평가 스크립트

실행:
    python eval/run_eval.py              # 전체 평가
    python eval/run_eval.py --category 추천/피부타입  # 카테고리별
    python eval/run_eval.py --id rec_001  # 단일 케이스

평가 지표:
    Hit@K  : 상위 K개 중 expected 제품이 1개 이상 → 1점
    MRR    : 첫 번째 정답의 순위 역수 평균
    정확도  : Hit@1 (1위 추천이 expected에 포함)
"""

import sys
import os
import json
import math
import argparse
from pathlib import Path
from collections import defaultdict

# 프로젝트 루트를 path에 추가
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.test_cases import TEST_CASES
from rag.retriever import (
    search_by_product_stats,
    search_by_skin_tone,
    normalize_skin_type,
    SKIN_TYPE_MAP,
    SKIN_TONE_MAP,
)


# ── 추천 결과 가져오기 ────────────────────────────────────────────────────────

def get_recommendation(case: dict) -> list[str]:
    """
    테스트 케이스 기반으로 추천 제품 goods_no 리스트 반환
    실제 챗봇 서브그래프 로직을 직접 호출
    """
    skin_type  = case.get("skin_type", "")
    skin_tone  = case.get("skin_tone", "")
    user_needs = case.get("user_needs", [])
    category   = case.get("category", "")

    skin_type_normalized = normalize_skin_type(skin_type)
    skin_tone_code       = SKIN_TONE_MAP.get(skin_tone, "")

    # 비교/필터 케이스: products_mentioned 그대로 반환
    if "비교" in category or "필터" in category:
        return case.get("products_mentioned", [])

    # 케이스 4: 피부톤만 있음 → skin_tone 기반 검색
    if not skin_type and skin_tone:
        docs = search_by_skin_tone(
            query=case["query"],
            skin_tone_code=skin_tone_code,
        )
        # 문서에서 제품 순서 추출 (중복 제거, 순서 유지)
        seen = []
        for doc in docs:
            gno = doc.metadata.get("goods_no", "")
            if gno and gno not in seen:
                seen.append(gno)
        return seen[:5]

    # 케이스 1,2: 피부타입 있음 → product_stats 스코어링
    return search_by_product_stats(
        user_needs=user_needs,
        skin_type=skin_type_normalized,
        skin_tone_code=skin_tone_code,
        top_k=5,
    )


# ── 지표 계산 ──────────────────────────────────────────────────────────────

def hit_at_k(predicted: list[str], expected: list[str], k: int) -> int:
    """Hit@K: 상위 K개 중 expected 포함 여부"""
    return int(bool(set(predicted[:k]) & set(expected)))


def reciprocal_rank(predicted: list[str], expected: list[str]) -> float:
    """MRR용 역수 순위"""
    for i, gno in enumerate(predicted):
        if gno in expected:
            return 1.0 / (i + 1)
    return 0.0


# ── 메인 평가 ──────────────────────────────────────────────────────────────

def run_eval(cases: list[dict], verbose: bool = True) -> dict:
    results = []
    category_scores = defaultdict(list)

    for case in cases:
        try:
            predicted = get_recommendation(case)
        except Exception as e:
            print(f"  ❌ [{case['id']}] 오류: {e}")
            predicted = []

        h1  = hit_at_k(predicted, case["expected_goods_nos"], 1)
        h3  = hit_at_k(predicted, case["expected_goods_nos"], 3)
        h5  = hit_at_k(predicted, case["expected_goods_nos"], 5)
        rr  = reciprocal_rank(predicted, case["expected_goods_nos"])

        result = {
            "id":        case["id"],
            "category":  case["category"],
            "query":     case["query"],
            "hit@1":     h1,
            "hit@3":     h3,
            "hit@5":     h5,
            "rr":        rr,
            "predicted": predicted[:3],
            "expected":  case["expected_goods_nos"],
            "desc":      case.get("description", ""),
        }
        results.append(result)
        category_scores[case["category"]].append(result)

        if verbose:
            status = "✅" if h3 else "❌"
            print(f"  {status} [{case['id']}] Hit@3={h3} RR={rr:.2f} | {case['query'][:35]}")
            if not h3:
                print(f"       예상: {case['expected_goods_nos'][:3]}")
                print(f"       실제: {predicted[:3]}")

    # 전체 지표
    n = len(results)
    overall = {
        "total":    n,
        "hit@1":    round(sum(r["hit@1"] for r in results) / n * 100, 1),
        "hit@3":    round(sum(r["hit@3"] for r in results) / n * 100, 1),
        "hit@5":    round(sum(r["hit@5"] for r in results) / n * 100, 1),
        "mrr":      round(sum(r["rr"]    for r in results) / n, 3),
    }

    # 카테고리별 지표
    by_category = {}
    for cat, cat_results in category_scores.items():
        m = len(cat_results)
        by_category[cat] = {
            "count": m,
            "hit@3": round(sum(r["hit@3"] for r in cat_results) / m * 100, 1),
            "mrr":   round(sum(r["rr"]    for r in cat_results) / m, 3),
        }

    return {"overall": overall, "by_category": by_category, "results": results}


# ── 결과 출력 ──────────────────────────────────────────────────────────────

def print_summary(eval_result: dict):
    overall = eval_result["overall"]
    print("\n" + "=" * 55)
    print("📊 평가 결과 요약")
    print("=" * 55)
    print(f"  총 케이스:  {overall['total']}개")
    print(f"  Hit@1:     {overall['hit@1']}%  (1위 추천 정확도)")
    print(f"  Hit@3:     {overall['hit@3']}%  (상위 3개 내 정답 포함)")
    print(f"  Hit@5:     {overall['hit@5']}%  (상위 5개 내 정답 포함)")
    print(f"  MRR:       {overall['mrr']}   (평균 역순위)")

    print("\n📁 카테고리별 Hit@3:")
    for cat, scores in eval_result["by_category"].items():
        bar = "█" * int(scores["hit@3"] / 10) + "░" * (10 - int(scores["hit@3"] / 10))
        print(f"  {cat:<20s} {bar} {scores['hit@3']:5.1f}%  ({scores['count']}건)")

    print("=" * 55)


def save_result(eval_result: dict, path: str = "eval/eval_result.json"):
    output_path = ROOT / path
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 상세 결과 저장 → {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="", help="카테고리 필터")
    parser.add_argument("--id",       type=str, default="", help="단일 케이스 ID")
    parser.add_argument("--quiet",    action="store_true",  help="케이스별 출력 숨김")
    args = parser.parse_args()

    # 필터 적용
    cases = TEST_CASES
    if args.id:
        cases = [c for c in cases if c["id"] == args.id]
    elif args.category:
        cases = [c for c in cases if args.category in c["category"]]

    if not cases:
        print("❌ 해당하는 케이스가 없어요.")
        sys.exit(1)

    print(f"\n🔍 평가 시작: {len(cases)}개 케이스\n")
    eval_result = run_eval(cases, verbose=not args.quiet)
    print_summary(eval_result)
    save_result(eval_result)