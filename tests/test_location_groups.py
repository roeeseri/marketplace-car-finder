from src.data.schema import CarAd
from src.location_groups import canonical_location, location_matches, location_region
from src.nlu.query_parser import parse_query
from src.search.filter import filter_ads


def test_parse_region_location():
    parsed = parse_query("רכב באזור מרכז עד 70 אלף")
    assert parsed.hard_constraints.location == "אזור מרכז"


def test_region_matching_in_filter():
    ads = [
        CarAd(make="טויוטה", model="יאריס", year=2020, price=50000, km=50000, location="תל אביב"),
        CarAd(make="טויוטה", model="יאריס", year=2020, price=50000, km=50000, location="אילת"),
    ]
    class Constraints:
        location = "אזור מרכז"
        make = model = price_max = price_min = year_min = year_max = km_max = gear_type = fuel_type = owners_max = None
    filtered = filter_ads(ads, Constraints())
    assert len(filtered) == 1
    assert filtered[0].location == "תל אביב"


def test_location_helpers():
    assert canonical_location("נהריה והסביבה") == "אזור נהריה והסביבה"
    assert location_region("חיפה") == "אזור נהריה והסביבה"
    assert location_matches("אזור מרכז", "רמת גן")
    assert not location_matches("אזור מרכז", "אילת")
