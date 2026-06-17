from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.data.schema import CarAd
from src.evaluation.baseline import baseline_rank_ads
from src.explanation.explainer import explain
from src.knowledge.vehicle_insights import build_insight_bundle
from src.nlu.query_parser import ParsedQuery, parse_query
from src.ranking.ranker import (
    RankedResult,
    RankingWeights,
    _hard_fit_score,
    _feature_match_score,
    _location_match_score,
    _vehicle_quality_score,
    rank,
)
from src.search.filter import filter_ads
from src.search.semantic_search import SemanticSearch


SearchStrategy = Literal["auto", "smart", "baseline"]


@dataclass(frozen=True)
class SearchDecision:
    strategy: Literal["smart", "baseline"]
    reason: str
    fallback_from: Literal["smart", "baseline"] | None = None
    fallback_reason: str | None = None


def _query_length(parsed: ParsedQuery) -> int:
    return len(parsed.raw_query.split())


def choose_search_strategy(parsed: ParsedQuery) -> SearchDecision:
    hard = parsed.hard_constraints
    soft = parsed.soft_preferences

    hard_slots = sum(
        1
        for value in [
            hard.make,
            hard.model,
            hard.price_max,
            hard.price_min,
            hard.year_min,
            hard.year_max,
            hard.km_max,
            hard.gear_type,
            hard.fuel_type,
            hard.location,
            hard.owners_max,
        ]
        if value is not None
    )
    soft_slots = sum(1 for value in soft.__dict__.values() if value)
    length = _query_length(parsed)

    baseline_score = 0.0
    smart_score = 0.0

    baseline_score += 1.4 if hard_slots >= 2 else 0.4
    baseline_score += 1.0 if hard_slots >= 4 else 0.2
    baseline_score += 0.6 if soft_slots == 0 else 0.0
    baseline_score += 0.4 if length <= 8 else 0.0

    smart_score += 1.4 if soft_slots >= 2 else 0.3
    smart_score += 0.8 if soft_slots >= 1 else 0.0
    smart_score += 0.6 if length >= 9 else 0.0
    smart_score += 0.4 if hard_slots <= 1 else 0.0

    if baseline_score >= smart_score:
        return SearchDecision(
            strategy="baseline",
            reason=(
                "השאילתה נראית קשיחה ומדויקת יותר, ולכן נבחר baseline "
                "שמסתדר טוב יותר עם תנאים מפורשים."
            ),
        )
    return SearchDecision(
        strategy="smart",
        reason=(
            "השאילתה כוללת כוונה סמנטית או העדפות רכות, ולכן נבחר smart "
            "כדי לנסות להבין את הכוונה המלאה."
        ),
    )


def _build_baseline_result(ad: CarAd, parsed: ParsedQuery, rank_index: int, total: int) -> RankedResult:
    position_score = max(0.0, 1.0 - (rank_index / max(total, 1)))
    quality_score = _vehicle_quality_score(ad)
    feature_score = _feature_match_score(ad.features, parsed.soft_preferences)
    location_score = _location_match_score(parsed.hard_constraints.location, ad.location)
    hard_fit_score = _hard_fit_score(ad, parsed)
    total_score = round(
        0.45 * position_score + 0.20 * quality_score + 0.15 * feature_score + 0.10 * min(location_score, 1.0) + 0.10 * hard_fit_score,
        4,
    )
    return RankedResult(
        ad=ad,
        total_score=total_score,
        semantic_score=0.0,
        vehicle_quality_score=round(quality_score, 4),
        feature_match_score=round(feature_score, 4),
        location_match_score=round(location_score, 4),
        hard_fit_score=round(hard_fit_score, 4),
    )


def _smart_search(
    ads: list[CarAd],
    searcher: SemanticSearch,
    parsed: ParsedQuery,
    top_n: int,
) -> list[tuple[RankedResult, str, object]]:
    hits = searcher.search(parsed.semantic_query, k=len(ads))
    ad_map = {ad.ad_id: ad for ad in ads}
    candidates = [(ad_map[h.ad_id], h.score) for h in hits if h.ad_id in ad_map]
    candidate_ads = [ad for ad, _ in candidates]
    allowed_ids = {a.ad_id for a in filter_ads(candidate_ads, parsed.hard_constraints)}
    filtered = [(ad, score) for ad, score in candidates if ad.ad_id in allowed_ids]
    if not filtered:
        return []
    ranked = rank(filtered, parsed, RankingWeights())[:top_n]
    return [
        (result, explain(result.ad, parsed, result), build_insight_bundle(result.ad, parsed, result))
        for result in ranked
    ]


def _baseline_search(
    ads: list[CarAd],
    parsed: ParsedQuery,
    top_n: int,
) -> list[tuple[RankedResult, str, object]]:
    ranked_ids = baseline_rank_ads(ads, parsed)
    ad_map = {ad.ad_id: ad for ad in ads}
    results: list[tuple[RankedResult, str, object]] = []
    for index, ad_id in enumerate(ranked_ids[:top_n], 1):
        ad = ad_map.get(ad_id)
        if ad is None:
            continue
        result = _build_baseline_result(ad, parsed, index - 1, len(ranked_ids[:top_n]))
        results.append((result, explain(ad, parsed, result), build_insight_bundle(ad, parsed, result)))
    return results


def route_search(
    query: str,
    ads: list[CarAd],
    searcher: SemanticSearch,
    top_n: int = 10,
    strategy: SearchStrategy = "auto",
) -> tuple[ParsedQuery, list[tuple[RankedResult, str, object]], SearchDecision]:
    parsed = parse_query(query)
    decision = choose_search_strategy(parsed) if strategy == "auto" else SearchDecision(
        strategy=strategy,
        reason="בחירה ידנית של המשתמש.",
    )

    if decision.strategy == "baseline":
        pairs = _baseline_search(ads, parsed, top_n)
        if not pairs:
            fallback_pairs = _smart_search(ads, searcher, parsed, top_n)
            if fallback_pairs:
                return (
                    parsed,
                    fallback_pairs,
                    SearchDecision(
                        strategy="smart",
                        reason=decision.reason,
                        fallback_from="baseline",
                        fallback_reason="baseline לא החזיר תוצאות, ולכן עברנו ל-smart כגיבוי.",
                    ),
                )
        return parsed, pairs, decision

    pairs = _smart_search(ads, searcher, parsed, top_n)
    if not pairs:
        fallback_pairs = _baseline_search(ads, parsed, top_n)
        if fallback_pairs:
            return (
                parsed,
                fallback_pairs,
                SearchDecision(
                    strategy="baseline",
                    reason=decision.reason,
                    fallback_from="smart",
                    fallback_reason="smart לא החזיר תוצאות, ולכן עברנו ל-baseline כגיבוי.",
                ),
            )
    return parsed, pairs, decision
