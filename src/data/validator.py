from dataclasses import dataclass
from typing import List

from config import (
    MAX_KM, MAX_PRICE, MAX_YEAR, MIN_KM,
    MIN_PRICE, MIN_YEAR, VALID_FUEL_TYPES, VALID_GEAR_TYPES,
)
from src.data.schema import CarAd


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]


def validate_ad(ad: CarAd) -> ValidationResult:
    errors = []

    if not ad.make or not str(ad.make).strip():
        errors.append("missing make")
    if not ad.model or not str(ad.model).strip():
        errors.append("missing model")

    try:
        year = int(ad.year)
        if not (MIN_YEAR <= year <= MAX_YEAR):
            errors.append(f"year {year} out of range [{MIN_YEAR}, {MAX_YEAR}]")
    except (TypeError, ValueError):
        errors.append(f"invalid year: {ad.year}")

    try:
        price = float(ad.price)
        if not (MIN_PRICE <= price <= MAX_PRICE):
            errors.append(f"price {price} out of range [{MIN_PRICE}, {MAX_PRICE}]")
    except (TypeError, ValueError):
        errors.append(f"invalid price: {ad.price}")

    try:
        km = float(ad.km)
        if not (MIN_KM <= km <= MAX_KM):
            errors.append(f"km {km} out of range [{MIN_KM}, {MAX_KM}]")
    except (TypeError, ValueError):
        errors.append(f"invalid km: {ad.km}")

    if ad.fuel_type and ad.fuel_type not in VALID_FUEL_TYPES:
        errors.append(f"unknown fuel_type: {ad.fuel_type}")

    if ad.gear_type and ad.gear_type not in VALID_GEAR_TYPES:
        errors.append(f"unknown gear_type: {ad.gear_type}")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


def validate_all(ads: List[CarAd]) -> tuple[List[CarAd], List[tuple[CarAd, List[str]]]]:
    """Returns (valid_ads, invalid_ads_with_errors)."""
    valid, invalid = [], []
    for ad in ads:
        result = validate_ad(ad)
        if result.is_valid:
            valid.append(ad)
        else:
            invalid.append((ad, result.errors))
    return valid, invalid
