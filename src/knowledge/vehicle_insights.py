from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional
from pathlib import Path

from config import BASE_DIR
from src.data.schema import CarAd
from src.location_groups import location_matches
from src.nlu.query_parser import ParsedQuery
from src.ranking.ranker import RankedResult


KNOWLEDGE_PATH = BASE_DIR / "data" / "knowledge" / "vehicle_sources.json"


@dataclass(frozen=True)
class MakeSource:
    make: str
    model: Optional[str]
    source_name: str
    source_url: str
    summary: str
    excerpt: str | None = None


@dataclass(frozen=True)
class VehicleInsight:
    class_label: str
    class_reason: str
    similarity_reasons: list[str]
    source: Optional[MakeSource]


@lru_cache(maxsize=1)
def _load_source_db() -> dict:
    if KNOWLEDGE_PATH.exists():
        with KNOWLEDGE_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {"sources": []}


def _normalize(value: str | None) -> str:
    return str(value).strip().lower() if value else ""


def _normalize_model(value: str | None) -> str:
    return _normalize(value).replace(" ", "")


def _model_candidates(make: str, model: str | None) -> list[str]:
    nm = _normalize_model(model)
    mk = _normalize(make)
    candidates = [nm] if nm else []

    if mk in {"ב.מ.וו", "bmw"} and nm:
        if nm.startswith(("320", "330", "318", "316", "325", "328", "335", "340")):
            candidates.append("3")
        if nm.startswith("x1"):
            candidates.append("x1")
    elif mk in {"יונדאי", "hyundai"} and nm:
        if nm.startswith("i20"):
            candidates.append("i20")
        if nm.startswith("i10"):
            candidates.append("i10")
    elif mk in {"טויוטה", "toyota"} and nm:
        if "auris" in nm:
            candidates.append("auris")
        if "yaris" in nm:
            candidates.append("yaris")
    elif mk in {"סוזוקי", "suzuki"} and nm:
        if "swift" in nm:
            candidates.append("swift")
    elif mk in {"מאזדה", "mazda"} and nm:
        if nm.startswith("2"):
            candidates.append("2")
    elif mk in {"ניסאן", "nissan"} and nm:
        if "micra" in nm:
            candidates.append("micra")
    elif mk in {"וולוו", "volvo"} and nm:
        if "xc40" in nm:
            candidates.append("xc40")
    elif mk in {"קיה", "kia"} and nm:
        if "picanto" in nm:
            candidates.append("picanto")

    return [item for idx, item in enumerate(candidates) if item and item not in candidates[:idx]]


def get_make_source(make: str | None, model: str | None = None) -> Optional[MakeSource]:
    normalized_make = _normalize(make)
    normalized_model = _normalize_model(model)
    if not normalized_make:
        return None

    db = _load_source_db()
    candidates = _model_candidates(make, model)
    for bucket in ("sources", "makes"):
        for row in db.get(bucket, []):
            if _normalize(row.get("make")) != normalized_make:
                continue
            row_model = _normalize_model(row.get("model"))
            if row_model and any(row_model == candidate for candidate in candidates):
                return MakeSource(
                    make=row.get("make", ""),
                    model=row.get("model"),
                    source_name=row.get("source_name", ""),
                    source_url=row.get("source_url", ""),
                    summary=row.get("summary", ""),
                    excerpt=row.get("excerpt"),
                )
    for row in db.get("makes", []):
        if _normalize(row.get("make")) == normalized_make:
            return MakeSource(
                make=row.get("make", ""),
                model=row.get("model"),
                source_name=row.get("source_name", ""),
                source_url=row.get("source_url", ""),
                summary=row.get("summary", ""),
                excerpt=row.get("excerpt"),
            )
    return None


_PREMIUM_MARKERS = {
    "ב.מ.וו",
    "מרצדס",
    "אודי",
    "וולוו",
    "לקסוס",
    "ג'נסיס",
}

_CITY_CAR_MARKERS = {
    "i10",
    "i20",
    "aygo",
    "swift",
    "rio",
    "micra",
    "yaris",
    "ibiza",
    "polo",
    "clio",
    "corsa",
    "208",
    "c3",
    "fiesta",
}

_SUV_MARKERS = {
    "x1",
    "x3",
    "x5",
    "tucson",
    "sportage",
    "sorento",
    "kodiaq",
    "rav4",
    "cx-5",
    "cx5",
    "qashqai",
    "crossover",
}

_FAMILY_MARKERS = {
    "sorento",
    "kodiaq",
    "outlander",
    "touring",
    "estate",
    "wagon",
    "cx-5",
    "tucson",
    "s-max",
    "grand",
}


def infer_vehicle_class(ad: CarAd) -> tuple[str, str]:
    text = " ".join(
        part for part in [
            ad.make,
            ad.model,
            ad.description or "",
            ad.fuel_type or "",
            ad.gear_type or "",
        ]
        if part
    ).lower()

    if any(marker.lower() in text for marker in _PREMIUM_MARKERS):
        return "פרימיום / מנהלים", "הדגם שייך למותג פרימיום"
    if any(marker in text for marker in _SUV_MARKERS):
        return "SUV / ג'יפון", "זוהה דגם שמיועד לנהיגה גבוהה וגמישה יותר"
    if any(marker in text for marker in _FAMILY_MARKERS):
        return "משפחתית", "הדגם מתאים יותר לשימוש משפחתי ויומיומי"
    if any(marker in text for marker in _CITY_CAR_MARKERS):
        return "קטן / עירוני", "הדגם קומפקטי ומתאים לעיר"
    if ad.fuel_type and "היבריד" in ad.fuel_type.lower():
        return "חסכוני / היברידי", "נרשם כמתאים להתנהלות חסכונית"
    return "כללי", "לא זוהתה קטגוריה ייחודית מובהקת"


def build_similarity_reasons(ad: CarAd, query: ParsedQuery, result: RankedResult | None = None) -> list[str]:
    reasons: list[str] = []
    vehicle_class, class_reason = infer_vehicle_class(ad)
    reasons.append(f"קטגוריה: {vehicle_class}")
    reasons.append(class_reason)

    hard = query.hard_constraints
    soft = query.soft_preferences

    if hard.price_max is not None and ad.price <= hard.price_max:
        reasons.append(f"נכנס לתקציב עד {int(hard.price_max):,} ₪")
    if hard.gear_type and ad.gear_type and hard.gear_type.lower() == ad.gear_type.lower():
        reasons.append("אותה תיבת הילוכים")
    if hard.fuel_type and ad.fuel_type and hard.fuel_type.lower() == ad.fuel_type.lower():
        reasons.append("אותו סוג דלק")
    if hard.location and ad.location and location_matches(hard.location, ad.location):
        reasons.append("קרוב לאזור המבוקש")
    if hard.make and ad.make and hard.make.lower() == ad.make.lower():
        reasons.append("אותו יצרן")

    if soft.fuel_efficient and ad.features and ad.features.fuel_efficient:
        reasons.append("חסכוני בדלק")
    if soft.family_car and ad.features and ad.features.family_car:
        reasons.append("מתאים למשפחה")
    if soft.first_owner and (ad.previous_owners == 1 or (ad.features and ad.features.first_owner)):
        reasons.append("יד ראשונה")
    if soft.luxury and ad.features and ad.features.luxury:
        reasons.append("יוקרתי")
    if soft.off_road and ad.features and ad.features.off_road:
        reasons.append("מתאים לשטח / ג'יפון")
    if soft.city_driving and vehicle_class in {"קטן / עירוני", "כללי"}:
        reasons.append("נוח לעיר")

    if result is not None:
        if result.location_match_score >= 1.5:
            reasons.append("התאמת מיקום חזקה")
        elif result.location_match_score > 0:
            reasons.append("התאמת מיקום חלקית")

    deduped: list[str] = []
    seen: set[str] = set()
    for item in reasons:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped[:6]


def build_explanation_summary(ad: CarAd, query: ParsedQuery, result: RankedResult) -> str:
    reasons = build_similarity_reasons(ad, query, result)
    if reasons:
        return "הרכב הומלץ מכיוון ש" + ", ".join(reasons) + "."
    return "הרכב הומלץ לפי התאמה כללית לשאילתה."


def build_insight_bundle(ad: CarAd, query: ParsedQuery, result: RankedResult) -> VehicleInsight:
    vehicle_class, class_reason = infer_vehicle_class(ad)
    return VehicleInsight(
        class_label=vehicle_class,
        class_reason=class_reason,
        similarity_reasons=build_similarity_reasons(ad, query, result),
        source=get_make_source(ad.make, ad.model),
    )
