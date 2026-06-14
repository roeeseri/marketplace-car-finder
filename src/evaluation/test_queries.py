from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    expected_constraints: Dict[str, object] = field(default_factory=dict)
    expected_soft_preferences: List[str] = field(default_factory=list)
    relevance: Dict[str, int] = field(default_factory=dict)


BENCHMARK_CASES: List[EvaluationCase] = [
    EvaluationCase(
        query="רכב ראשון לסטודנט עד 50 אלף, אוטומטי, חסכוני",
        expected_constraints={"price_max": 50000, "gear_type": "אוטומטי"},
        expected_soft_preferences=["fuel_efficient", "young_driver", "first_owner"],
        relevance={"122": 3, "188": 3, "235": 3, "296": 3, "301": 2, "1": 2, "2": 2, "6": 2, "9": 2, "11": 2, "16": 2, "25": 2, "32": 2},
    ),
    EvaluationCase(
        query="משפחתית מרווחת אוטומטית עד 80 אלף",
        expected_constraints={"price_max": 80000, "gear_type": "אוטומטי"},
        expected_soft_preferences=["family_car"],
        relevance={"287": 3, "238": 3, "121": 3, "86": 2, "218": 2, "122": 2, "188": 2, "256": 2, "251": 2, "301": 1},
    ),
    EvaluationCase(
        query="ג'יפ 4x4 אוטומטי עד 150 אלף",
        expected_constraints={"price_max": 150000, "gear_type": "אוטומטי"},
        expected_soft_preferences=["off_road", "family_car"],
        relevance={"5": 3, "397": 3, "505": 2},
    ),
    EvaluationCase(
        query="BMW אוטומטי עד 150 אלף",
        expected_constraints={"make": "ב.מ.וו", "price_max": 150000, "gear_type": "אוטומטי"},
        expected_soft_preferences=["luxury"],
        relevance={"357": 3, "463": 3, "340": 2, "396": 2, "412": 2, "309": 1, "311": 1},
    ),
    EvaluationCase(
        query="רכב חסכוני בדלק עד 60 אלף",
        expected_constraints={"price_max": 60000},
        expected_soft_preferences=["fuel_efficient"],
        relevance={"1": 3, "2": 3, "6": 3, "9": 3, "16": 3, "32": 3, "122": 2, "183": 2, "236": 2, "256": 2, "296": 2},
    ),
    EvaluationCase(
        query="רכב עירוני לנהג צעיר עד 45 אלף",
        expected_constraints={"price_max": 45000},
        expected_soft_preferences=["young_driver", "city_driving"],
        relevance={"122": 3, "183": 3, "235": 3, "296": 3, "1": 2, "2": 2, "9": 2, "11": 2, "25": 2, "32": 2, "64": 2, "67": 2, "76": 2},
    ),
    EvaluationCase(
        query="רכב יוקרתי עם טסט ארוך",
        expected_constraints={},
        expected_soft_preferences=["luxury"],
        relevance={"307": 3, "308": 3, "309": 3, "311": 3, "312": 3, "313": 3, "317": 3, "318": 3, "319": 3, "340": 2, "357": 2, "463": 2},
    ),
    EvaluationCase(
        query="רכב יד ראשונה ללא תאונות",
        expected_constraints={"owners_max": 1},
        expected_soft_preferences=["first_owner", "reliable"],
        relevance={"1": 3, "2": 3, "6": 3, "9": 3, "13": 3, "16": 3, "17": 3, "32": 3, "34": 3, "46": 3, "70": 2, "122": 2, "188": 2, "235": 2, "296": 2},
    ),
    EvaluationCase(
        query="טויוטה היברידית חסכונית יד ראשונה עד 35 אלף",
        expected_constraints={"make": "טויוטה", "price_max": 35000, "fuel_type": "היברידי", "owners_max": 1},
        expected_soft_preferences=["fuel_efficient", "first_owner"],
        relevance={"211": 3, "50": 2, "167": 2},
    ),
    EvaluationCase(
        query="ב.מ.וו X1 יוקרתי אוטומטי יד ראשונה עד 180 אלף",
        expected_constraints={"make": "ב.מ.וו", "model": "X1", "price_max": 180000, "gear_type": "אוטומטי", "owners_max": 1},
        expected_soft_preferences=["luxury", "first_owner"],
        relevance={"379": 3, "396": 3, "412": 3, "358": 2, "481": 2},
    ),
    EvaluationCase(
        query="רכב קטן לעיר לסטודנט עד 50 אלף",
        expected_constraints={"price_max": 50000},
        expected_soft_preferences=["young_driver", "city_driving"],
        relevance={"6": 3, "16": 3, "18": 3, "22": 3, "30": 3, "96": 2, "106": 2, "128": 2, "148": 2, "176": 2, "183": 2},
    ),
    EvaluationCase(
        query="רכב ידני חסכוני עד 60 אלף",
        expected_constraints={"price_max": 60000, "gear_type": "ידני"},
        expected_soft_preferences=["fuel_efficient"],
        relevance={"3": 3, "18": 3, "22": 3, "30": 3, "43": 3, "96": 2, "106": 2, "128": 2, "148": 2, "163": 2, "176": 2, "183": 2, "192": 2},
    ),
    EvaluationCase(
        query="מרצדס אוטומטית יוקרתית עד 180 אלף",
        expected_constraints={"make": "מרצדס", "price_max": 180000, "gear_type": "אוטומטי"},
        expected_soft_preferences=["luxury"],
        relevance={"313": 3, "359": 3, "405": 3, "448": 3, "463": 3, "419": 2, "435": 2},
    ),
]


def case_by_query(query: str) -> Optional[EvaluationCase]:
    for case in BENCHMARK_CASES:
        if case.query == query:
            return case
    return None
