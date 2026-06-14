from __future__ import annotations

import random
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from config import SAMPLE_CSV
from src.data.feature_extractor import extract_all
from src.data.loader import load_from_csv
from src.data.preprocessor import preprocess_all
from src.data.validator import validate_all
from src.explanation.explainer import explain
from src.nlu.query_parser import parse_query
from src.ranking.ranker import RankingWeights, rank
from src.search.embedder import Embedder
from src.search.filter import filter_ads
from src.search.index_builder import build_faiss_index, build_id_map
from src.search.semantic_search import SemanticSearch

from .baseline import baseline_rank_ads
from .metrics import MetricSummary, ndcg_at_k, precision_at_k, recall_at_k, slot_precision_recall_f1, summarise
from .test_queries import BENCHMARK_CASES, EvaluationCase
from .llm_judge import judge_relevance


@dataclass(frozen=True)
class VariantSummary:
    name: str
    summary: MetricSummary


@dataclass(frozen=True)
class TimingSummary:
    name: str
    mean_ms: float
    median_ms: float
    peak_kb: float


def _hard_prediction(parsed) -> Dict[str, object]:
    return {
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
    }


def _soft_prediction(parsed) -> Dict[str, object]:
    return {name: True for name, enabled in parsed.soft_preferences.__dict__.items() if enabled}


def _soft_expected(case: EvaluationCase) -> Dict[str, object]:
    return {name: True for name in case.expected_soft_preferences}


def _slot_summary(scores: Sequence[Dict[str, float]]) -> Dict[str, float]:
    return {
        "precision": statistics.mean(item["precision"] for item in scores),
        "recall": statistics.mean(item["recall"] for item in scores),
        "f1": statistics.mean(item["f1"] for item in scores),
    }


def load_and_prepare_ads(csv_path=SAMPLE_CSV):
    ads = load_from_csv(csv_path)
    ads = preprocess_all(ads)
    valid_ads, _ = validate_all(ads)
    valid_ads = extract_all(valid_ads)
    return valid_ads


def build_searcher(ads) -> SemanticSearch:
    embedder = Embedder()
    embeddings = embedder.encode_ads(ads)
    index = build_faiss_index(embeddings)
    id_map = build_id_map(ads)
    return SemanticSearch(index, id_map, embedder)


def _smart_rank(ads, searcher: SemanticSearch, case: EvaluationCase, k: int) -> List[str]:
    parsed = parse_query(case.query)
    hits = searcher.search(parsed.semantic_query, k=len(ads))
    ad_map = {str(ad.ad_id): ad for ad in ads}
    candidates = [(ad_map[h.ad_id], h.score) for h in hits if h.ad_id in ad_map]
    candidate_ads = [ad for ad, _ in candidates]
    filtered_ids = {a.ad_id for a in filter_ads(candidate_ads, parsed.hard_constraints)}
    filtered = [(ad, score) for ad, score in candidates if ad.ad_id in filtered_ids]
    if not filtered:
        return []
    ranked = rank(filtered, parsed, RankingWeights())[:k]
    return [str(r.ad.ad_id) for r in ranked]


def _semantic_only_rank(ads, searcher: SemanticSearch, case: EvaluationCase, k: int) -> List[str]:
    parsed = parse_query(case.query)
    hits = searcher.search(parsed.semantic_query, k=len(ads))
    ad_map = {str(ad.ad_id): ad for ad in ads}
    candidate_ids = [h.ad_id for h in hits if h.ad_id in ad_map]
    candidate_ads = [ad_map[ad_id] for ad_id in candidate_ids]
    filtered = filter_ads(candidate_ads, parsed.hard_constraints)
    return [str(ad.ad_id) for ad in filtered[:k]]


def _relevance_scores(case: EvaluationCase, ranked_ids: Sequence[str], k: int) -> List[int]:
    return [case.relevance.get(ad_id, 0) for ad_id in ranked_ids[:k]]


def evaluate_nlu(cases: Sequence[EvaluationCase] = BENCHMARK_CASES) -> Dict[str, float]:
    hard_scores = []
    soft_scores = []
    for case in cases:
        parsed = parse_query(case.query)
        hard_scores.append(slot_precision_recall_f1(predicted=_hard_prediction(parsed), expected=case.expected_constraints))
        soft_scores.append(slot_precision_recall_f1(predicted=_soft_prediction(parsed), expected=_soft_expected(case)))
    hard = _slot_summary(hard_scores)
    soft = _slot_summary(soft_scores)
    combined = {
        "precision": (hard["precision"] + soft["precision"]) / 2,
        "recall": (hard["recall"] + soft["recall"]) / 2,
        "f1": (hard["f1"] + soft["f1"]) / 2,
    }
    return {
        "precision": hard["precision"],
        "recall": hard["recall"],
        "f1": hard["f1"],
        "hard": hard,
        "soft": soft,
        "combined": combined,
    }


def evaluate_retrieval_variants(ads, cases: Sequence[EvaluationCase] = BENCHMARK_CASES, k: int = 5):
    searcher = build_searcher(ads)
    smart_rankings = []
    baseline_rankings = []
    semantic_only_rankings = []

    for case in cases:
        smart_ids = _smart_rank(ads, searcher, case, k)
        baseline_ids = baseline_rank_ads(ads, parse_query(case.query))[:k]
        semantic_ids = _semantic_only_rank(ads, searcher, case, k)
        smart_rankings.append(_relevance_scores(case, smart_ids, k))
        baseline_rankings.append(_relevance_scores(case, baseline_ids, k))
        semantic_only_rankings.append(_relevance_scores(case, semantic_ids, k))

    return {
        "smart": summarise(smart_rankings, k),
        "baseline": summarise(baseline_rankings, k),
        "no_rerank": summarise(semantic_only_rankings, k),
    }


def evaluate_case_rows(ads, cases: Sequence[EvaluationCase] = BENCHMARK_CASES, k: int = 5):
    searcher = build_searcher(ads)
    rows = []
    for case in cases:
        parsed = parse_query(case.query)
        smart_ids = _smart_rank(ads, searcher, case, k)
        baseline_ids = baseline_rank_ads(ads, parsed)[:k]
        semantic_ids = _semantic_only_rank(ads, searcher, case, k)

        smart_rel = _relevance_scores(case, smart_ids, k)
        baseline_rel = _relevance_scores(case, baseline_ids, k)
        semantic_rel = _relevance_scores(case, semantic_ids, k)
        total_relevant = sum(1 for rel in case.relevance.values() if rel > 0)

        rows.append({
            "query": case.query,
            "nlu_hard_precision": slot_precision_recall_f1(predicted=_hard_prediction(parsed), expected=case.expected_constraints)["precision"],
            "nlu_hard_recall": slot_precision_recall_f1(predicted=_hard_prediction(parsed), expected=case.expected_constraints)["recall"],
            "nlu_hard_f1": slot_precision_recall_f1(predicted=_hard_prediction(parsed), expected=case.expected_constraints)["f1"],
            "nlu_soft_precision": slot_precision_recall_f1(predicted=_soft_prediction(parsed), expected=_soft_expected(case))["precision"],
            "nlu_soft_recall": slot_precision_recall_f1(predicted=_soft_prediction(parsed), expected=_soft_expected(case))["recall"],
            "nlu_soft_f1": slot_precision_recall_f1(predicted=_soft_prediction(parsed), expected=_soft_expected(case))["f1"],
            "nlu_combined_f1": (
                slot_precision_recall_f1(predicted=_hard_prediction(parsed), expected=case.expected_constraints)["f1"]
                + slot_precision_recall_f1(predicted=_soft_prediction(parsed), expected=_soft_expected(case))["f1"]
            ) / 2,
            "smart_p@5": precision_at_k(smart_rel, k),
            "smart_r@5": recall_at_k(smart_rel, k, total_relevant),
            "smart_ndcg@5": ndcg_at_k(smart_rel, k),
            "baseline_p@5": precision_at_k(baseline_rel, k),
            "baseline_r@5": recall_at_k(baseline_rel, k, total_relevant),
            "baseline_ndcg@5": ndcg_at_k(baseline_rel, k),
            "no_rerank_p@5": precision_at_k(semantic_rel, k),
            "no_rerank_r@5": recall_at_k(semantic_rel, k, total_relevant),
            "no_rerank_ndcg@5": ndcg_at_k(semantic_rel, k),
            "smart_top5": ", ".join(smart_ids),
            "baseline_top5": ", ".join(baseline_ids),
            "no_rerank_top5": ", ".join(semantic_ids),
        })
    return rows


def evaluate_timings(ads, cases: Sequence[EvaluationCase] = BENCHMARK_CASES, repeats: int = 3):
    searcher = build_searcher(ads)
    timings: Dict[str, List[float]] = {"smart": [], "baseline": []}
    peaks: Dict[str, List[float]] = {"smart": [], "baseline": []}

    for _ in range(repeats):
        for case in cases:
            parsed = parse_query(case.query)

            tracemalloc.start()
            t0 = time.perf_counter()
            _ = _smart_rank(ads, searcher, case, k=5)
            smart_elapsed = (time.perf_counter() - t0) * 1000
            _, smart_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            timings["smart"].append(smart_elapsed)
            peaks["smart"].append(smart_peak / 1024)

            tracemalloc.start()
            t0 = time.perf_counter()
            _ = baseline_rank_ads(ads, parsed)
            baseline_elapsed = (time.perf_counter() - t0) * 1000
            _, baseline_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            timings["baseline"].append(baseline_elapsed)
            peaks["baseline"].append(baseline_peak / 1024)

    return {
        "smart": TimingSummary(
            name="smart",
            mean_ms=statistics.mean(timings["smart"]),
            median_ms=statistics.median(timings["smart"]),
            peak_kb=statistics.mean(peaks["smart"]),
        ),
        "baseline": TimingSummary(
            name="baseline",
            mean_ms=statistics.mean(timings["baseline"]),
            median_ms=statistics.median(timings["baseline"]),
            peak_kb=statistics.mean(peaks["baseline"]),
        ),
    }


def evaluate_judge_sample(ads, cases: Sequence[EvaluationCase] = BENCHMARK_CASES, sample_size: int = 50, k: int = 5):
    rng = random.Random(42)
    all_pairs = [(case, ad) for case in cases for ad in ads]
    sampled = all_pairs if len(all_pairs) <= sample_size else rng.sample(all_pairs, sample_size)
    judged = []
    for case, ad in sampled:
        result = judge_relevance(parse_query(case.query), ad)
        judged.append({
            "query": case.query,
            "ad_id": str(ad.ad_id),
            "score": result.score,
            "rationale": result.rationale,
            "make": ad.make,
            "model": ad.model,
        })

    return judged


def evaluate_ablation(ads, cases: Sequence[EvaluationCase] = BENCHMARK_CASES, k: int = 5):
    searcher = build_searcher(ads)
    no_semantic_rankings = []
    no_rerank_rankings = []

    for case in cases:
        parsed = parse_query(case.query)
        no_semantic = baseline_rank_ads(ads, parsed)[:k]
        no_semantic_rankings.append(_relevance_scores(case, no_semantic, k))
        no_rerank = _semantic_only_rank(ads, searcher, case, k)
        no_rerank_rankings.append(_relevance_scores(case, no_rerank, k))

    return {
        "no_semantic": summarise(no_semantic_rankings, k),
        "no_rerank": summarise(no_rerank_rankings, k),
    }


def build_full_report(csv_path=SAMPLE_CSV, cases: Sequence[EvaluationCase] = BENCHMARK_CASES, k: int = 5):
    ads = load_and_prepare_ads(csv_path)
    nlu = evaluate_nlu(cases)
    retrieval = evaluate_retrieval_variants(ads, cases, k=k)
    case_rows = evaluate_case_rows(ads, cases, k=k)
    ablation = evaluate_ablation(ads, cases, k=k)
    timings = evaluate_timings(ads, cases)
    judge_sample = evaluate_judge_sample(ads, cases)
    return {
        "nlu": nlu,
        "retrieval": retrieval,
        "case_rows": case_rows,
        "ablation": ablation,
        "timings": timings,
        "judge_sample": judge_sample,
    }


def _fmt_summary(summary: MetricSummary) -> str:
    return f"P@{5}={summary.precision_at_k:.3f} R@{5}={summary.recall_at_k:.3f} NDCG@{5}={summary.ndcg_at_k:.3f}"


def print_full_report(report: Dict[str, object]) -> None:
    print("NLU")
    print(f"  hard: precision={report['nlu']['hard']['precision']:.3f} recall={report['nlu']['hard']['recall']:.3f} f1={report['nlu']['hard']['f1']:.3f}")
    print(f"  soft: precision={report['nlu']['soft']['precision']:.3f} recall={report['nlu']['soft']['recall']:.3f} f1={report['nlu']['soft']['f1']:.3f}")
    print(f"  combined: precision={report['nlu']['combined']['precision']:.3f} recall={report['nlu']['combined']['recall']:.3f} f1={report['nlu']['combined']['f1']:.3f}")
    print()

    print("Retrieval")
    for name, summary in report["retrieval"].items():
        print(f"  {name}: {_fmt_summary(summary)}")
    print()

    print("Ablation")
    for name, summary in report["ablation"].items():
        print(f"  {name}: {_fmt_summary(summary)}")
    print()

    print("Time / Memory")
    for name, timing in report["timings"].items():
        print(f"  {name}: mean={timing.mean_ms:.2f}ms median={timing.median_ms:.2f}ms peak={timing.peak_kb:.1f}KB")
    print()

    print("Judge sample")
    hist = {}
    for item in report["judge_sample"]:
        hist[item["score"]] = hist.get(item["score"], 0) + 1
    print(f"  sampled_pairs={len(report['judge_sample'])} histogram={hist}")


def main() -> None:
    report = build_full_report()
    print_full_report(report)


if __name__ == "__main__":
    main()
