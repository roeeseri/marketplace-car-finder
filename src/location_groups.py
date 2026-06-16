from __future__ import annotations

from typing import Optional

REGION_CITY_MAP = {
    "\u05d0\u05d6\u05d5\u05e8 \u05de\u05e8\u05db\u05d6": {
        "\u05ea\u05dc \u05d0\u05d1\u05d9\u05d1",
        "\u05e8\u05de\u05ea \u05d2\u05df",
        "\u05d2\u05d1\u05e2\u05ea\u05d9\u05d9\u05dd",
        "\u05d7\u05d5\u05dc\u05d5\u05df",
        "\u05d1\u05ea \u05d9\u05dd",
        "\u05e8\u05d0\u05e9\u05d5\u05df \u05dc\u05e6\u05d9\u05d5\u05df",
        "\u05e0\u05e1 \u05e6\u05d9\u05d5\u05e0\u05d4",
        "\u05e8\u05d7\u05d5\u05d1\u05d5\u05ea",
        "\u05d9\u05d1\u05e0\u05d4",
        "\u05e4\u05ea\u05d7 \u05ea\u05e7\u05d5\u05d5\u05d4",
        "\u05e7\u05e8\u05d9\u05d9\u05ea \u05d0\u05d5\u05e0\u05d5",
        "\u05de\u05d5\u05d3\u05d9\u05e2\u05d9\u05df",
        "\u05d4\u05e8\u05e6\u05dc\u05d9\u05d4",
        "\u05e8\u05e2\u05e0\u05e0\u05d4",
        "\u05db\u05e4\u05e8 \u05e1\u05d1\u05d0",
        "\u05e0\u05ea\u05e0\u05d9\u05d4",
    },
    "\u05d0\u05d6\u05d5\u05e8 \u05e0\u05d4\u05e8\u05d9\u05d4 \u05d5\u05d4\u05e1\u05d1\u05d9\u05d1\u05d4": {
        "\u05e0\u05d4\u05e8\u05d9\u05d4",
        "\u05d7\u05d9\u05e4\u05d4",
        "\u05e7\u05e8\u05d9\u05d5\u05ea",
        "\u05e2\u05e4\u05d5\u05dc\u05d4",
    },
    "\u05d0\u05d6\u05d5\u05e8 \u05d9\u05e8\u05d5\u05e9\u05dc\u05d9\u05dd": {
        "\u05d9\u05e8\u05d5\u05e9\u05dc\u05d9\u05dd",
    },
    "\u05d0\u05d6\u05d5\u05e8 \u05d3\u05e8\u05d5\u05dd": {
        "\u05d1\u05d0\u05e8 \u05e9\u05d1\u05e2",
        "\u05d0\u05e9\u05d3\u05d5\u05d3",
        "\u05d0\u05e9\u05e7\u05dc\u05d5\u05df",
        "\u05d0\u05d9\u05dc\u05ea",
    },
}

LOCATION_REGION_ALIASES = {
    "\u05d0\u05d6\u05d5\u05e8 \u05de\u05e8\u05db\u05d6": "\u05d0\u05d6\u05d5\u05e8 \u05de\u05e8\u05db\u05d6",
    "\u05de\u05e8\u05db\u05d6": "\u05d0\u05d6\u05d5\u05e8 \u05de\u05e8\u05db\u05d6",
    "\u05de\u05e8\u05db\u05d6 \u05d4\u05d0\u05e8\u05e5": "\u05d0\u05d6\u05d5\u05e8 \u05de\u05e8\u05db\u05d6",
    "\u05d0\u05d6\u05d5\u05e8 \u05e0\u05d4\u05e8\u05d9\u05d4 \u05d5\u05d4\u05e1\u05d1\u05d9\u05d1\u05d4": "\u05d0\u05d6\u05d5\u05e8 \u05e0\u05d4\u05e8\u05d9\u05d4 \u05d5\u05d4\u05e1\u05d1\u05d9\u05d1\u05d4",
    "\u05e0\u05d4\u05e8\u05d9\u05d4 \u05d5\u05d4\u05e1\u05d1\u05d9\u05d1\u05d4": "\u05d0\u05d6\u05d5\u05e8 \u05e0\u05d4\u05e8\u05d9\u05d4 \u05d5\u05d4\u05e1\u05d1\u05d9\u05d1\u05d4",
    "\u05d0\u05d6\u05d5\u05e8 \u05e0\u05d4\u05e8\u05d9\u05d4": "\u05d0\u05d6\u05d5\u05e8 \u05e0\u05d4\u05e8\u05d9\u05d4 \u05d5\u05d4\u05e1\u05d1\u05d9\u05d1\u05d4",
    "\u05e6\u05e4\u05d5\u05df": "\u05d0\u05d6\u05d5\u05e8 \u05e0\u05d4\u05e8\u05d9\u05d4 \u05d5\u05d4\u05e1\u05d1\u05d9\u05d1\u05d4",
    "\u05d0\u05d6\u05d5\u05e8 \u05e6\u05e4\u05d5\u05df": "\u05d0\u05d6\u05d5\u05e8 \u05e0\u05d4\u05e8\u05d9\u05d4 \u05d5\u05d4\u05e1\u05d1\u05d9\u05d1\u05d4",
    "\u05d0\u05d6\u05d5\u05e8 \u05d9\u05e8\u05d5\u05e9\u05dc\u05d9\u05dd": "\u05d0\u05d6\u05d5\u05e8 \u05d9\u05e8\u05d5\u05e9\u05dc\u05d9\u05dd",
    "\u05d9\u05e8\u05d5\u05e9\u05dc\u05d9\u05dd \u05d5\u05d4\u05e1\u05d1\u05d9\u05d1\u05d4": "\u05d0\u05d6\u05d5\u05e8 \u05d9\u05e8\u05d5\u05e9\u05dc\u05d9\u05dd",
    "\u05d0\u05d6\u05d5\u05e8 \u05d3\u05e8\u05d5\u05dd": "\u05d0\u05d6\u05d5\u05e8 \u05d3\u05e8\u05d5\u05dd",
    "\u05d3\u05e8\u05d5\u05dd": "\u05d0\u05d6\u05d5\u05e8 \u05d3\u05e8\u05d5\u05dd",
}

CITY_TO_REGION = {
    city: region
    for region, cities in REGION_CITY_MAP.items()
    for city in cities
}

KNOWN_LOCATION_TOKENS = set(LOCATION_REGION_ALIASES) | set(CITY_TO_REGION)


def _normalize(text: str | None) -> str:
    return str(text).strip() if text else ""


def extract_location(value: str | None) -> Optional[str]:
    text = _normalize(value)
    if not text:
        return None

    for alias, canonical in sorted(LOCATION_REGION_ALIASES.items(), key=lambda item: -len(item[0])):
        if alias in text:
            return canonical

    for city in sorted(CITY_TO_REGION.keys(), key=len, reverse=True):
        if city in text:
            return city

    return None


def canonical_location(value: str | None) -> Optional[str]:
    text = _normalize(value)
    if not text:
        return None
    found = extract_location(text)
    return found or text


def location_region(value: str | None) -> Optional[str]:
    canonical = canonical_location(value)
    if not canonical:
        return None
    if canonical in REGION_CITY_MAP:
        return canonical
    return CITY_TO_REGION.get(canonical)


def location_matches(query_location: str | None, ad_location: str | None) -> bool:
    q = canonical_location(query_location)
    a = canonical_location(ad_location)
    if not q:
        return True
    if not a:
        return False

    if q in REGION_CITY_MAP:
        return location_region(a) == q or a in REGION_CITY_MAP[q]
    return q == a or location_region(a) == location_region(q)
