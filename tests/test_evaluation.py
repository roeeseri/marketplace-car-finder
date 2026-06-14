from config import SAMPLE_CSV
from src.data.loader import load_from_csv
from src.data.preprocessor import preprocess_all
from src.data.validator import validate_all
from src.evaluation.baseline import evaluate_benchmark
from src.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k, slot_precision_recall_f1
from src.evaluation.test_queries import BENCHMARK_CASES


def _load_ads():
    ads = load_from_csv(SAMPLE_CSV)
    ads = preprocess_all(ads)
    valid_ads, invalid_ads = validate_all(ads)
    assert not invalid_ads
    return valid_ads


def test_metrics_basic():
    relevances = [3, 0, 2, 1, 0]
    assert precision_at_k(relevances, 3) == 2 / 3
    assert recall_at_k(relevances, 3, 3) == 2 / 3
    assert 0 < ndcg_at_k(relevances, 3) <= 1


def test_slot_scores_exact_match():
    scores = slot_precision_recall_f1({"price_max": 50000, "gear_type": "אוטומטי"}, {"price_max": 50000, "gear_type": "אוטומטי"})
    assert scores == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_benchmark_runs():
    ads = _load_ads()
    results, summary = evaluate_benchmark(ads, BENCHMARK_CASES, k=5)
    assert len(results) == len(BENCHMARK_CASES)
    assert summary.n_cases == len(BENCHMARK_CASES)
    assert 0 <= summary.precision_at_k <= 1
    assert 0 <= summary.recall_at_k <= 1
    assert 0 <= summary.ndcg_at_k <= 1
