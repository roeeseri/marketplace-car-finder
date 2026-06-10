from src.data.schema import CarAd
from src.nlu.query_parser import ParsedQuery
from src.ranking.ranker import RankedResult


def explain(ad: CarAd, query: ParsedQuery, result: RankedResult) -> str:
    parts = []
    c = query.hard_constraints
    f = ad.features

    if c.price_max is not None:
        parts.append(f"עומד בתקציב ({int(ad.price):,} ₪)")

    if c.gear_type and ad.gear_type:
        parts.append(f"תיבת הילוכים {ad.gear_type}")

    if c.owners_max == 1 or (f and f.first_owner):
        parts.append("יד ראשונה")

    if f and f.fuel_efficient:
        parts.append("חסכוני בדלק")

    if f and f.authorized_garage:
        parts.append("מטופל במוסך מורשה")

    if f and f.accident_free:
        parts.append("ללא תאונות")

    if f and f.family_car:
        parts.append("מתאים למשפחה")

    if f and f.off_road:
        parts.append("כלי רכב שטח")

    parts.append(f"שנת {ad.year}")
    parts.append(f"{int(ad.km):,} ק\"מ")

    return "הרכב הומלץ מכיוון ש" + ", ".join(parts) + "."
