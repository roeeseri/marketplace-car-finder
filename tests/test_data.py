import pytest

from src.data.preprocessor import normalize_km, normalize_price, normalize_text, preprocess_ad
from src.data.schema import CarAd
from src.data.validator import validate_ad, validate_all


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_ad(**kwargs) -> CarAd:
    defaults = dict(make="טויוטה", model="יאריס", year=2019, price=38000, km=62000)
    defaults.update(kwargs)
    return CarAd(**defaults)


# ── Schema ────────────────────────────────────────────────────────────────────

def test_car_ad_from_dict_roundtrip():
    data = {"make": "מאזדה", "model": "3", "year": "2018", "price": "42000",
            "km": "85000", "gear_type": "אוטומטי", "fuel_type": "בנזין",
            "ad_id": "1", "engine_volume": "1500", "previous_owners": "1",
            "location": "תל אביב", "description": "רכב שמור"}
    ad = CarAd.from_dict(data)
    assert ad.make == "מאזדה"
    assert ad.year == 2018
    assert ad.price == 42000.0
    assert ad.to_dict()["model"] == "3"


# ── Preprocessor ─────────────────────────────────────────────────────────────

def test_normalize_price_plain():
    assert normalize_price("42000") == 42000.0

def test_normalize_price_hebrew_shorthand():
    assert normalize_price("42 אלף") == 42000.0

def test_normalize_price_with_commas():
    assert normalize_price("42,000") == 42000.0

def test_normalize_km():
    assert normalize_km("85,000") == 85000.0

def test_normalize_text_strips_whitespace():
    assert normalize_text("  רכב  שמור  ") == "רכב שמור"

def test_normalize_text_none():
    assert normalize_text(None) == ""

def test_preprocess_ad_normalizes_price():
    ad = make_ad(price="45 אלף")
    processed = preprocess_ad(ad)
    assert processed.price == 45000.0


# ── Validator ─────────────────────────────────────────────────────────────────

def test_valid_ad_passes():
    result = validate_ad(make_ad())
    assert result.is_valid
    assert result.errors == []

def test_missing_make_fails():
    result = validate_ad(make_ad(make=""))
    assert not result.is_valid
    assert any("make" in e for e in result.errors)

def test_year_out_of_range_fails():
    result = validate_ad(make_ad(year=1850))
    assert not result.is_valid

def test_price_out_of_range_fails():
    result = validate_ad(make_ad(price=-1))
    assert not result.is_valid

def test_invalid_gear_type_fails():
    result = validate_ad(make_ad(gear_type="turbo"))
    assert not result.is_valid

def test_validate_all_splits_correctly():
    ads = [make_ad(), make_ad(make=""), make_ad(year=1800)]
    valid, invalid = validate_all(ads)
    assert len(valid) == 1
    assert len(invalid) == 2
