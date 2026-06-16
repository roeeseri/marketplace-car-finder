from src.data.schema import CarAd
from src.knowledge.vehicle_insights import build_insight_bundle, get_make_source, infer_vehicle_class
from src.nlu.query_parser import parse_query
from src.ranking.ranker import RankedResult


def _ranked(ad: CarAd) -> RankedResult:
    return RankedResult(
        ad=ad,
        total_score=1.0,
        semantic_score=0.8,
        vehicle_quality_score=0.7,
        feature_match_score=0.6,
        location_match_score=1.0,
    )


def test_bmw_source_lookup():
    source = get_make_source("ב.מ.וו")
    assert source is not None
    assert source.source_url.startswith("https://")
    assert "BMW" in source.source_name


def test_infer_vehicle_class_city_car():
    ad = CarAd(make="יונדאי", model="i20", year=2024, price=62000, km=12000)
    label, reason = infer_vehicle_class(ad)
    assert label == "קטן / עירוני"
    assert "עיר" in reason


def test_similarity_bundle_contains_source_and_reasons():
    query = parse_query("BMW אוטומטי עד 150 אלף")
    ad = CarAd(make="ב.מ.וו", model="320i", year=2022, price=145000, km=42000, gear_type="אוטומטי")
    bundle = build_insight_bundle(ad, query, _ranked(ad))
    assert bundle.source is not None
    assert bundle.source.source_name
    assert any("אותו יצרן" in reason for reason in bundle.similarity_reasons)
    assert any("תיבת הילוכים" in reason for reason in bundle.similarity_reasons)
