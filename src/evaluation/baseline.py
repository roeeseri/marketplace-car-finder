from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from src.data.loader import load_from_csv
from src.data.preprocessor import preprocess_all
from src.data.validator import validate_all
from src.nlu.query_parser import parse_query
from src.search.filter import filter_ads

from .metrics import MetricSummary, precision_at_k, recall_at_k, ndcg_at_k, slot_precision_recall_f1, summarise
from .test_queries import BENCHMARK_CASES, EvaluationCase


@dataclass(frozen=True)
class CaseResult:
    query: str
    slot_scores: Dict[str, float]
    relevance_at_k: List[int]
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    ranked_ad_ids: List[str]


def baseline_rank_ads(ads, parsed_query) -> List[str]:
    filtered_ads = filter_ads(ads, parsed_query.hard_constraints)
    ranked = sorted(
        filtered_ads,
        key=lambda ad: (
            -ad.year,
            ad.price,
            ad.km,
            -(ad.previous_owners or 1),
            str(ad.ad_id),
        ),
    )
    return [str(ad.ad_id) for ad in ranked]


def evaluate_case(case: EvaluationCase, ads, k: int = 5) -> CaseResult:
    parsed = parse_query(case.query)
    slot_scores = slot_precision_recall_f1(
        predicted={
            "make": parsed.hard_constraints.make,
            "model": parsed.hard_constraints.model,
            "price_max": parsed.hard_constraints.price_max,
            "price_min": parsed.hard_constraints.price_min,
            "year_min": parsed.hard_constraints.year_min,
            "year_max": parsed.hard_constraints.year_max,
            "km_max": parsed.hard_constraints.km_max,
            "gear_type": parsed.hard_constraints.gear_type,
            "fuel_type": parsed.hard_constraints.fuel_type,
            "location": parsed.hard_constraints.location,
            "owners_max": parsed.hard_constraints.owners_max,
        },
        expected=case.expected_constraints,
    )

    ranked_ids = baseline_rank_ads(ads, parsed)
    relevance_at_k = [case.relevance.get(ad_id, 0) for ad_id in ranked_ids[:k]]
    return CaseResult(
        query=case.query,
        slot_scores=slot_scores,
        relevance_at_k=relevance_at_k,
        precision_at_k=precision_at_k(relevance_at_k, k),
        recall_at_k=recall_at_k(relevance_at_k, k, sum(1 for rel in case.relevance.values() if rel > 0)),
        ndcg_at_k=ndcg_at_k(relevance_at_k, k),
        ranked_ad_ids=ranked_ids[:k],
    )


def evaluate_benchmark(ads, cases: Sequence[EvaluationCase] = BENCHMARK_CASES, k: int = 5):
    results = [evaluate_case(case, ads, k=k) for case in cases]
    summary = summarise([result.relevance_at_k for result in results], k=k)
    return results, summary


def load_and_prepare_ads(csv_path):
    ads = load_from_csv(csv_path)
    ads = preprocess_all(ads)
    valid_ads, _ = validate_all(ads)
    return valid_ads


def main() -> None:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    from config import SAMPLE_CSV

    ads = load_and_prepare_ads(SAMPLE_CSV)
    results, summary = evaluate_benchmark(ads)

    print("Evaluation summary")
    print(f"cases: {summary.n_cases}")
    print(f"precision@5: {summary.precision_at_k:.3f}")
    print(f"recall@5: {summary.recall_at_k:.3f}")
    print(f"ndcg@5: {summary.ndcg_at_k:.3f}")
    print()
    for result in results:
        print(f"- {result.query}")
        print(f"  slots: P={result.slot_scores['precision']:.3f} R={result.slot_scores['recall']:.3f} F1={result.slot_scores['f1']:.3f}")
        print(f"  top5: {result.ranked_ad_ids}")
        print(f"  rank metrics: P@5={result.precision_at_k:.3f} R@5={result.recall_at_k:.3f} NDCG@5={result.ndcg_at_k:.3f}")


if __name__ == "__main__":
    main()
