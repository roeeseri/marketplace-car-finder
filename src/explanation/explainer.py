from src.data.schema import CarAd
from src.knowledge.vehicle_insights import build_explanation_summary
from src.nlu.query_parser import ParsedQuery
from src.ranking.ranker import RankedResult


def explain(ad: CarAd, query: ParsedQuery, result: RankedResult) -> str:
    return build_explanation_summary(ad, query, result)
