from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class MetricSummary:
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    n_cases: int


def precision_at_k(relevances: Sequence[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = list(relevances[:k])
    return sum(1 for r in top_k if r > 0) / k


def recall_at_k(relevances: Sequence[int], k: int, total_relevant: int) -> float:
    if total_relevant <= 0:
        return 0.0
    top_k = list(relevances[:k])
    return sum(1 for r in top_k if r > 0) / total_relevant


def dcg_at_k(relevances: Sequence[int], k: int) -> float:
    score = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        if rel <= 0:
            continue
        score += (2**rel - 1) / math.log2(i + 1)
    return score


def ndcg_at_k(relevances: Sequence[int], k: int) -> float:
    actual = dcg_at_k(relevances, k)
    ideal = dcg_at_k(sorted(relevances, reverse=True), k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def slot_precision_recall_f1(
    predicted: Dict[str, object],
    expected: Dict[str, object],
) -> Dict[str, float]:
    def _norm(value: object) -> object:
        if isinstance(value, str):
            return value.strip().casefold()
        return value

    predicted_items = {k: v for k, v in predicted.items() if v is not None}
    expected_items = {k: v for k, v in expected.items() if v is not None}

    if not predicted_items and not expected_items:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    correct = sum(1 for k, v in predicted_items.items() if _norm(expected_items.get(k)) == _norm(v))
    precision = correct / len(predicted_items) if predicted_items else 0.0
    recall = correct / len(expected_items) if expected_items else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def summarise(rankings: Iterable[Sequence[int]], k: int) -> MetricSummary:
    rankings = list(rankings)
    if not rankings:
        return MetricSummary(0.0, 0.0, 0.0, 0)

    precisions = [precision_at_k(r, k) for r in rankings]
    recalls = []
    ndcgs = []
    for relevances in rankings:
        total_relevant = sum(1 for r in relevances if r > 0)
        recalls.append(recall_at_k(relevances, k, total_relevant))
        ndcgs.append(ndcg_at_k(relevances, k))

    return MetricSummary(
        precision_at_k=sum(precisions) / len(precisions),
        recall_at_k=sum(recalls) / len(recalls),
        ndcg_at_k=sum(ndcgs) / len(ndcgs),
        n_cases=len(rankings),
    )
