import re
from typing import List

import pandas as pd

from src.data.schema import CarAd


def _clean_number_string(value: str) -> str:
    """Remove thousands separators and Hebrew shorthand (e.g. '45 אלף' → '45000')."""
    value = str(value).strip()
    value = re.sub(r"[‏‎]", "", value)  # remove RTL/LTR marks
    value = value.replace(",", "").replace("'", "")
    match = re.match(r"(\d+(?:\.\d+)?)\s*אלף", value)
    if match:
        return str(float(match.group(1)) * 1000)
    return value


def normalize_price(value: str | float | int) -> float:
    return float(_clean_number_string(str(value)))


def normalize_km(value: str | float | int) -> float:
    return float(_clean_number_string(str(value)))


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_gear_type(value: str | None) -> str | None:
    if not value:
        return None
    mapping = {
        "automatic": "אוטומטי",
        "manual": "ידני",
        "auto": "אוטומטי",
    }
    normalized = str(value).strip().lower()
    return mapping.get(normalized, str(value).strip())


def preprocess_ad(ad: CarAd) -> CarAd:
    ad.price = normalize_price(ad.price)
    ad.km = normalize_km(ad.km)
    ad.description = normalize_text(ad.description)
    ad.gear_type = normalize_gear_type(ad.gear_type)
    if ad.make:
        ad.make = str(ad.make).strip()
    if ad.model:
        ad.model = str(ad.model).strip()
    if ad.location:
        ad.location = str(ad.location).strip()
    return ad


def preprocess_all(ads: List[CarAd]) -> List[CarAd]:
    return [preprocess_ad(ad) for ad in ads]


def to_dataframe(ads: List[CarAd]) -> pd.DataFrame:
    return pd.DataFrame([ad.to_dict() for ad in ads])
