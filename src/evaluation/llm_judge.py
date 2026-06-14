from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from src.data.schema import CarAd
from src.nlu.query_parser import ParsedQuery


@dataclass(frozen=True)
class JudgeResult:
    score: int
    rationale: str


def build_judge_prompt(query: str, ad: CarAd) -> str:
    return (
        "You are grading how well a car listing matches a user request on a 0-3 scale.\n"
        f"Query: {query}\n"
        f"Ad: {ad.make} {ad.model} {ad.year}, price={ad.price}, km={ad.km}, "
        f"gear={ad.gear_type}, fuel={ad.fuel_type}, location={ad.location}\n"
        f"Description: {ad.description or ''}\n"
        "Return only a score and a short rationale."
    )


def judge_relevance(query: ParsedQuery, ad: CarAd) -> JudgeResult:
    hard = query.hard_constraints
    reasons = []

    if hard.make and ad.make and hard.make.lower() != ad.make.lower():
        return JudgeResult(0, "make mismatch")
    if hard.model and ad.model and hard.model.lower() != ad.model.lower():
        return JudgeResult(0, "model mismatch")
    if hard.price_max is not None and ad.price > hard.price_max:
        return JudgeResult(0, "price too high")
    if hard.price_min is not None and ad.price < hard.price_min:
        return JudgeResult(0, "price too low")
    if hard.year_min is not None and ad.year < hard.year_min:
        return JudgeResult(0, "year too old")
    if hard.year_max is not None and ad.year > hard.year_max:
        return JudgeResult(0, "year too new")
    if hard.km_max is not None and ad.km > hard.km_max:
        return JudgeResult(0, "km too high")
    if hard.gear_type and ad.gear_type and hard.gear_type.lower() != ad.gear_type.lower():
        return JudgeResult(0, "gear mismatch")
    if hard.fuel_type and ad.fuel_type and hard.fuel_type.lower() != ad.fuel_type.lower():
        return JudgeResult(0, "fuel mismatch")
    if hard.location and ad.location and hard.location.lower() != ad.location.lower():
        return JudgeResult(0, "location mismatch")
    if hard.owners_max is not None and ad.previous_owners is not None and ad.previous_owners > hard.owners_max:
        return JudgeResult(0, "too many owners")

    description = (ad.description or "").lower()
    soft_hits = 0
    if query.soft_preferences.family_car and any(term in description for term in ["משפח", "מרווח", "7 מושבים"]):
        soft_hits += 1
    if query.soft_preferences.fuel_efficient and "חסכ" in description:
        soft_hits += 1
    if query.soft_preferences.first_owner and ("יד ראשונה" in description or ad.previous_owners == 1):
        soft_hits += 1
    if query.soft_preferences.luxury and any(term in description for term in ["יוקר", "פרימיום"]):
        soft_hits += 1
    if query.soft_preferences.off_road and any(term in description for term in ["4x4", "שטח", "ג'יפ"]):
        soft_hits += 1

    if soft_hits >= 3:
        return JudgeResult(3, "hard constraints satisfied and strong soft preference match")
    if soft_hits == 2:
        return JudgeResult(2, "hard constraints satisfied with solid soft match")
    if soft_hits == 1:
        return JudgeResult(1, "hard constraints satisfied with limited soft match")
    return JudgeResult(1, "hard constraints satisfied")
