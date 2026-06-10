from dataclasses import dataclass
from typing import List, Tuple

from config import MIN_YEAR, MAX_YEAR
from src.data.schema import CarAd, CarFeatures
from src.nlu.query_parser import ParsedQuery

_MAX_KM     = 250_000
_MAX_OWNERS = 5


@dataclass
class RankingWeights:
    semantic:        float = 0.40
    vehicle_quality: float = 0.35
    feature_match:   float = 0.25


@dataclass
class RankedResult:
    ad:                    CarAd
    total_score:           float
    semantic_score:        float
    vehicle_quality_score: float
    feature_match_score:   float


def rank(
    candidates: List[Tuple[CarAd, float]],
    query: ParsedQuery,
    weights: RankingWeights = None,
) -> List[RankedResult]:
    if weights is None:
        weights = RankingWeights()
    results = [_score(ad, sem, query, weights) for ad, sem in candidates]
    results.sort(key=lambda r: r.total_score, reverse=True)
    return results


def _score(
    ad: CarAd,
    semantic_score: float,
    query: ParsedQuery,
    w: RankingWeights,
) -> RankedResult:
    vq    = _vehicle_quality_score(ad)
    fm    = _feature_match_score(ad.features, query.soft_preferences)
    total = w.semantic * semantic_score + w.vehicle_quality * vq + w.feature_match * fm
    return RankedResult(
        ad=ad,
        total_score=round(total, 4),
        semantic_score=round(semantic_score, 4),
        vehicle_quality_score=round(vq, 4),
        feature_match_score=round(fm, 4),
    )


def _vehicle_quality_score(ad: CarAd) -> float:
    year_score  = (ad.year - MIN_YEAR) / max(MAX_YEAR - MIN_YEAR, 1)
    km_score    = max(0.0, 1.0 - ad.km / _MAX_KM)
    owner_score = max(0.0, 1.0 - (ad.previous_owners or 1) / _MAX_OWNERS)
    return round((year_score + km_score + owner_score) / 3, 4)


def _feature_match_score(features: CarFeatures, prefs) -> float:
    if features is None:
        return 0.5
    mapping = {
        "fuel_efficient": features.fuel_efficient,
        "family_car":     features.family_car,
        "first_owner":    features.first_owner,
        "luxury":         features.luxury,
        "off_road":       features.off_road,
    }
    requested = [k for k, v in prefs.__dict__.items() if v and k in mapping]
    if not requested:
        return 0.5
    matched = sum(1 for k in requested if mapping.get(k) is True)
    return matched / len(requested)
