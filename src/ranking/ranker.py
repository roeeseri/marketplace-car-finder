from dataclasses import dataclass
from typing import List, Tuple

from config import MIN_YEAR, MAX_YEAR
from src.data.schema import CarAd, CarFeatures
from src.location_groups import REGION_CITY_MAP, canonical_location, location_region
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
    location_match_score:  float
    hard_fit_score:        float = 0.0


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
    lm    = _location_match_score(query.hard_constraints.location, ad.location)
    hf    = _hard_fit_score(ad, query)
    total = 0.30 * semantic_score + 0.20 * vq + 0.12 * fm + 0.20 * hf + 0.18 * lm
    return RankedResult(
        ad=ad,
        total_score=round(total, 4),
        semantic_score=round(semantic_score, 4),
        vehicle_quality_score=round(vq, 4),
        feature_match_score=round(fm, 4),
        location_match_score=round(lm, 4),
        hard_fit_score=round(hf, 4),
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


def _location_match_score(query_location: str | None, ad_location: str | None) -> float:
    q = canonical_location(query_location)
    a = canonical_location(ad_location)
    if not q or not a:
        return 0.0
    q_region = location_region(q)
    a_region = location_region(a)
    if q == a:
        return 2.0 if q_region and q_region not in {q} else 1.5
    if q_region and a_region and q_region == a_region:
        return 0.7 if q not in REGION_CITY_MAP else 1.0
    return 0.0


def _hard_fit_score(ad: CarAd, query: ParsedQuery) -> float:
    hard = query.hard_constraints
    scores: list[float] = []

    if hard.make:
        scores.append(1.0 if ad.make and hard.make.lower() == ad.make.lower() else 0.0)
    if hard.model:
        scores.append(1.0 if ad.model and hard.model.lower() == ad.model.lower() else 0.0)
    if hard.price_max is not None and hard.price_max > 0:
        scores.append(max(0.0, min(1.0, 1.0 - (ad.price / hard.price_max))))
    if hard.price_min is not None and hard.price_min > 0:
        scores.append(max(0.0, min(1.0, ad.price / hard.price_min)))
    if hard.year_min is not None:
        denom = max(MAX_YEAR - hard.year_min, 1)
        scores.append(max(0.0, min(1.0, (ad.year - hard.year_min) / denom)))
    if hard.year_max is not None:
        denom = max(hard.year_max - MIN_YEAR, 1)
        scores.append(max(0.0, min(1.0, (hard.year_max - ad.year) / denom)))
    if hard.km_max is not None and hard.km_max > 0:
        scores.append(max(0.0, min(1.0, 1.0 - (ad.km / hard.km_max))))
    if hard.gear_type:
        scores.append(1.0 if ad.gear_type and hard.gear_type.lower() == ad.gear_type.lower() else 0.0)
    if hard.fuel_type:
        scores.append(1.0 if ad.fuel_type and hard.fuel_type.lower() == ad.fuel_type.lower() else 0.0)
    if hard.location:
        scores.append(_location_match_score(hard.location, ad.location) / 2.0)
    if hard.owners_max is not None:
        if ad.previous_owners is None:
            scores.append(0.5)
        else:
            scores.append(max(0.0, min(1.0, 1.0 - (ad.previous_owners / max(hard.owners_max, 1)))))

    if not scores:
        return 0.5
    return sum(scores) / len(scores)
