from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable

from src.data.schema import CarAd
from src.knowledge.vehicle_insights import infer_vehicle_class
from src.location_groups import location_matches


def _normalize(value: str | None) -> str:
    return str(value).strip().lower() if value else ""


def _same_text(a: str | None, b: str | None) -> bool:
    return bool(_normalize(a)) and _normalize(a) == _normalize(b)


@dataclass(frozen=True)
class ValuationResult:
    estimated_price: int
    low_price: int
    high_price: int
    comparable_count: int
    basis: str
    confidence: float
    reasons: list[str]
    comparable_preview: list[str]
    histogram_labels: list[str]
    histogram_counts: list[int]


def _year_factor(target_year: int, comp_year: int) -> float:
    diff = target_year - comp_year
    return max(0.65, min(1.55, 1.0 + diff * 0.045))


def _km_factor(target_km: float, comp_km: float) -> float:
    diff = comp_km - target_km
    return max(0.72, min(1.35, 1.0 + (diff / 100_000.0) * 0.12))


def _owners_factor(target_owners: int | None, comp_owners: int | None) -> float:
    target = target_owners or 2
    comp = comp_owners or 2
    diff = comp - target
    if diff > 0:
        return min(1.12, 1.0 + 0.03 * diff)
    if diff < 0:
        return max(0.90, 1.0 - 0.025 * abs(diff))
    return 1.0


def _gear_factor(target_gear: str | None, comp_gear: str | None) -> float:
    if _same_text(target_gear, comp_gear):
        return 1.0
    if "אוטומ" in _normalize(target_gear) and "ידני" in _normalize(comp_gear):
        return 1.04
    if "ידני" in _normalize(target_gear) and "אוטומ" in _normalize(comp_gear):
        return 0.96
    return 1.0


def _fuel_factor(target_fuel: str | None, comp_fuel: str | None) -> float:
    if _same_text(target_fuel, comp_fuel):
        return 1.0
    t = _normalize(target_fuel)
    c = _normalize(comp_fuel)
    if not t or not c:
        return 1.0
    if "היבריד" in t and "היבריד" not in c:
        return 1.05
    if "חשמל" in t and "חשמל" not in c:
        return 1.08
    if "דיזל" in t and "בנזין" in c:
        return 0.98
    return 1.0


def _location_factor(target_location: str | None, comp_location: str | None) -> float:
    if not target_location or not comp_location:
        return 1.0
    return 1.015 if location_matches(target_location, comp_location) else 1.0


def _condition_factor(target_condition: str | None) -> float:
    c = _normalize(target_condition)
    if not c:
        return 1.0
    if any(term in c for term in ["מעולה", "מצוין", "חדש"]):
        return 1.10
    if any(term in c for term in ["טוב", "שמור"]):
        return 1.0
    if any(term in c for term in ["סביר", "בסדר", "רגיל"]):
        return 0.92
    if any(term in c for term in ["דורש", "השקעה", "פגום"]):
        return 0.82
    return 1.0


def _target_fallback_model(ad: CarAd) -> str:
    return _normalize(ad.model)


def _comparable_score(target: CarAd, comp: CarAd, same_class: bool) -> float:
    score = 0.0
    if _same_text(target.make, comp.make):
        score += 2.0
    if _same_text(target.model, comp.model):
        score += 2.5
    elif _target_fallback_model(target) and _target_fallback_model(target) in _normalize(comp.model):
        score += 1.2
    if same_class:
        score += 1.0
    score += max(0.0, 1.0 - abs(target.year - comp.year) / 12.0)
    score += max(0.0, 1.0 - abs(target.km - comp.km) / 200_000.0)
    if _same_text(target.gear_type, comp.gear_type):
        score += 0.25
    if _same_text(target.fuel_type, comp.fuel_type):
        score += 0.25
    if target.location and comp.location and location_matches(target.location, comp.location):
        score += 0.15
    if target.previous_owners is not None and comp.previous_owners is not None and target.previous_owners == comp.previous_owners:
        score += 0.1
    return score


def _adjusted_price(target: CarAd, comp: CarAd, target_condition: str | None = None) -> float:
    price = float(comp.price)
    price *= _year_factor(target.year, comp.year)
    price *= _km_factor(target.km, comp.km)
    price *= _owners_factor(target.previous_owners, comp.previous_owners)
    price *= _gear_factor(target.gear_type, comp.gear_type)
    price *= _fuel_factor(target.fuel_type, comp.fuel_type)
    price *= _location_factor(target.location, comp.location)
    price *= _condition_factor(target_condition)
    if _same_text(target.make, comp.make):
        price *= 1.02
    return price


def _group_label(target: CarAd, comparable_count: int, exact_model_count: int, exact_make_count: int) -> str:
    if exact_model_count >= 3:
        return "אותו דגם"
    if exact_make_count >= 5:
        return "אותו יצרן"
    if comparable_count >= 8:
        return infer_vehicle_class(target)[0]
    return "מאגר דומים כללי"


def estimate_car_value(target: CarAd, ads: Iterable[CarAd], target_condition: str | None = None) -> ValuationResult:
    ads = [ad for ad in ads if ad.price and ad.year]
    exact_model = [ad for ad in ads if _same_text(ad.make, target.make) and _same_text(ad.model, target.model)]
    exact_make = [ad for ad in ads if _same_text(ad.make, target.make)]
    same_class = [ad for ad in ads if infer_vehicle_class(ad)[0] == infer_vehicle_class(target)[0]]

    if len(exact_model) >= 3:
        pool = exact_model
    elif len(exact_make) >= 5:
        pool = exact_make
    elif len(same_class) >= 8:
        pool = same_class
    else:
        pool = ads

    scored = sorted(
        ((ad, _comparable_score(target, ad, infer_vehicle_class(ad)[0] == infer_vehicle_class(target)[0])) for ad in pool),
        key=lambda item: item[1],
        reverse=True,
    )
    if not scored:
        return ValuationResult(
            estimated_price=0,
            low_price=0,
            high_price=0,
            comparable_count=0,
            basis="אין מספיק דאטה",
            confidence=0.0,
            reasons=["לא נמצאו רכבים להשוואה"],
            comparable_preview=[],
            histogram_labels=[],
            histogram_counts=[],
        )

    top = [ad for ad, _ in scored[: max(8, min(20, len(scored)))]]
    adjusted = [_adjusted_price(target, ad, target_condition) for ad in top]
    median_price = statistics.median(adjusted)
    low_price = statistics.quantiles(adjusted, n=4, method="inclusive")[0] if len(adjusted) >= 4 else min(adjusted)
    high_price = statistics.quantiles(adjusted, n=4, method="inclusive")[2] if len(adjusted) >= 4 else max(adjusted)

    reasons = [
        f"השוואה ל-{len(top)} רכבים דומים",
        f"בסיס לפי {_group_label(target, len(scored), len(exact_model), len(exact_make))}",
    ]
    if target_condition:
        reasons.append(f"מצב הרכב הוגדר כ-{target_condition}")
    reasons.append("תיקון לפי שנתון, קילומטראז' ומספר בעלים")
    if target.location:
        reasons.append("שקלול עדין של התאמת מיקום")

    preview = [f"{ad.make} {ad.model} {ad.year} - {int(ad.price):,} ₪" for ad, _ in scored[:3]]

    if len(adjusted) >= 2:
        bucket_min = min(adjusted)
        bucket_max = max(adjusted)
        span = max(bucket_max - bucket_min, 1)
        edges = [bucket_min + span * i / 5 for i in range(6)]
        counts = [0, 0, 0, 0, 0]
        for value in adjusted:
            idx = min(4, int((value - bucket_min) / span * 5))
            counts[idx] += 1
    else:
        counts = [len(adjusted), 0, 0, 0, 0]
        edges = [adjusted[0]] * 6

    histogram_labels = [
        f"{int(round(edges[i])):,}–{int(round(edges[i + 1])):,} ₪" if edges[i] != edges[i + 1] else f"{int(round(edges[i])):,} ₪"
        for i in range(5)
    ]

    spread = max(high_price - low_price, 1)
    confidence = max(0.35, min(0.92, 0.45 + min(len(top), 20) / 40 + min(1.0, spread / max(median_price, 1)) * 0.05))

    return ValuationResult(
        estimated_price=int(round(median_price)),
        low_price=int(round(low_price)),
        high_price=int(round(high_price)),
        comparable_count=len(top),
        basis=_group_label(target, len(scored), len(exact_model), len(exact_make)),
        confidence=round(confidence, 2),
        reasons=reasons,
        comparable_preview=preview,
        histogram_labels=histogram_labels,
        histogram_counts=counts,
    )
