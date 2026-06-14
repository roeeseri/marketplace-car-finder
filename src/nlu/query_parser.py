import re
from dataclasses import dataclass
from typing import Optional

# ── Known makes (Hebrew variants + English → canonical Hebrew) ────────────────

_MAKES = {
    "מאזדה": "מאזדה", "mazda": "מאזדה",
    "טויוטה": "טויוטה", "toyota": "טויוטה",
    "יונדאי": "יונדאי", "hyundai": "יונדאי",
    "קיה": "קיה", "kia": "קיה",
    "סקודה": "סקודה", "skoda": "סקודה",
    "פולקסווגן": "פולקסווגן", "volkswagen": "פולקסווגן", "vw": "פולקסווגן",
    "פורד": "פורד", "ford": "פורד",
    "ניסן": "ניסן", "nissan": "ניסן",
    "הונדה": "הונדה", "honda": "הונדה",
    "מיצובישי": "מיצובישי", "mitsubishi": "מיצובישי",
    "סוזוקי": "סוזוקי", "suzuki": "סוזוקי",
    "רנו": "רנו", "renault": "רנו",
    "סיאט": "סיאט", "seat": "סיאט",
    "אאודי": "אאודי", "audi": "אאודי",
    "מרצדס": "מרצדס", "mercedes": "מרצדס",
    "ב.מ.וו": "ב.מ.וו", "bmw": "ב.מ.וו",
    "וולוו": "וולוו", "volvo": "וולוו",
    "אופל": "אופל", "opel": "אופל",
    "סיטרואן": "סיטרואן", "citroen": "סיטרואן",
    "פיאט": "פיאט", "fiat": "פיאט",
    "שברולט": "שברולט", "chevrolet": "שברולט",
    "לקסוס": "לקסוס", "lexus": "לקסוס",
}

_LOCATIONS = [
    "תל אביב", "ירושלים", "חיפה", "באר שבע", "ראשון לציון",
    "פתח תקווה", "אשדוד", "נתניה", "רמת גן", "גבעתיים",
    "הרצליה", "כפר סבא", "רעננה", "מודיעין", "חולון",
    "בת ים", "אילת", "טבריה", "נצרת", "רחובות",
    "לוד", "רמלה", "נס ציונה", "יבנה", "אשקלון",
    "צפון", "דרום", "מרכז", "שרון", "שפלה", "גליל", "נגב",
]

_SOFT_KEYWORDS = {
    "family_car":     ["משפחתי", "למשפחה", "מרווח", "מרחבי", "מינוואן"],
    "fuel_efficient": ["חסכוני", "חסכנות", "צריכה נמוכה", "חסכון בדלק"],
    "reliable":       ["אמין", "אמינות", "אמינה", "עמיד", "מהימן"],
    "first_owner":    ["יד ראשונה", "יד 1", "בעלים ראשון"],
    "young_driver":   ["סטודנט", "נהג צעיר", "נהגת צעירה", "רישיון חדש", "רכב ראשון", "לסטודנט"],
    "off_road":       ["שטח", "4x4", "4X4", "כונן כפול", "ג'יפ", "ג׳יפ"],
    "luxury":         ["יוקרה", "פרימיום", "לוקסוס", "יוקרתי"],
    "city_driving":   ["עירוני", "לעיר", "פרקינג"],
    "long_trips":     ["נסיעות ארוכות", "בין עירוני", "בין-עירוני", "לנסיעות"],
}

# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class HardConstraints:
    make:       Optional[str]   = None
    model:      Optional[str]   = None
    price_max:  Optional[float] = None
    price_min:  Optional[float] = None
    year_min:   Optional[int]   = None
    year_max:   Optional[int]   = None
    km_max:     Optional[float] = None
    gear_type:  Optional[str]   = None
    fuel_type:  Optional[str]   = None
    location:   Optional[str]   = None
    owners_max: Optional[int]   = None


@dataclass
class SoftPreferences:
    family_car:     bool = False
    fuel_efficient: bool = False
    reliable:       bool = False
    first_owner:    bool = False
    young_driver:   bool = False
    off_road:       bool = False
    luxury:         bool = False
    city_driving:   bool = False
    long_trips:     bool = False


@dataclass
class ParsedQuery:
    raw_query:        str
    hard_constraints: HardConstraints
    soft_preferences: SoftPreferences
    semantic_query:   str
    language:         str = "he"

# ── Extraction helpers ────────────────────────────────────────────────────────

def _extract_price_max(text: str) -> Optional[float]:
    m = re.search(r'עד\s*(\d+(?:,\d+)?)\s*אלף', text)
    if m:
        return float(m.group(1).replace(",", "")) * 1000
    m = re.search(r'תקציב[^,\d]*(\d+)\s*אלף', text)
    if m:
        return float(m.group(1)) * 1000
    m = re.search(r'(?:עד|מקסימום)\s*(\d{4,7})\b', text)
    if m:
        return float(m.group(1))
    return None


def _extract_km_max(text: str) -> Optional[float]:
    m = re.search(r'עד\s*(\d+(?:,\d+)?)\s*אלף\s*ק["״]?מ', text)
    if m:
        return float(m.group(1).replace(",", "")) * 1000
    m = re.search(r'עד\s*(\d{4,7})\s*ק["״]?מ', text)
    if m:
        return float(m.group(1))
    return None


def _extract_year_range(text: str) -> tuple:
    m = re.search(r'(20\d{2})\s*[-–]\s*(20\d{2})', text)
    if m:
        return int(m.group(1)), int(m.group(2))
    year_min, year_max = None, None
    m = re.search(r'מ(?:שנת)?\s*(20\d{2})', text)
    if m:
        year_min = int(m.group(1))
    m = re.search(r'עד\s*שנת?\s*(20\d{2})', text)
    if m:
        year_max = int(m.group(1))
    if not year_min and not year_max:
        m = re.search(r'\b(20\d{2})\b', text)
        if m:
            year_min = int(m.group(1))
    return year_min, year_max


def _extract_make_model(text: str) -> tuple:
    make, model = None, None
    for variant, canonical in sorted(_MAKES.items(), key=lambda x: -len(x[0])):
        pattern = r'(?<!\w)' + re.escape(variant) + r'(?!\w)'
        if re.search(pattern, text, re.IGNORECASE):
            make = canonical
            idx = text.lower().find(variant.lower())
            after = text[idx + len(variant):]
            m = re.search(r'^\s+(\S+)', after)
            if m:
                candidate = m.group(1).strip()
                _stop = {"עד", "מ", "ב", "ל", "ו", "של", "עם", "אלף",
                         "שנה", "אוטומטי", "אוטומטית", "ידני", "ידנית",
                         "דיזל", "בנזין", "היברידי", "היברידית",
                         "יוקרתי", "יוקרתית", "חסכוני", "חסכונית",
                         "עירוני", "עירונית", "סטודנט", "לסטודנט",
                         "משפחתי", "משפחתית", "רכב", "ראשונה"}
                if candidate and candidate not in _stop and len(candidate) <= 12:
                    model = candidate
            break
    return make, model


def _extract_gear_type(text: str) -> Optional[str]:
    if re.search(r'אוטומט(י)?|automatic|auto\b', text, re.IGNORECASE):
        return "אוטומטי"
    if re.search(r'ידני|manual', text, re.IGNORECASE):
        return "ידני"
    return None


def _extract_fuel_type(text: str) -> Optional[str]:
    if re.search(r'היברידי|hybrid', text, re.IGNORECASE):
        return "היברידי"
    if re.search(r'חשמלי|electric', text, re.IGNORECASE):
        return "חשמלי"
    if re.search(r'דיזל|diesel', text, re.IGNORECASE):
        return "דיזל"
    if re.search(r'\bגז\b|lpg', text, re.IGNORECASE):
        return "גז"
    return None


def _extract_location(text: str) -> Optional[str]:
    for loc in sorted(_LOCATIONS, key=len, reverse=True):
        if loc in text:
            return loc
    return None


def _extract_owners_max(text: str) -> Optional[int]:
    if re.search(r'יד ראשונה|יד 1|בעלים ראשון', text):
        return 1
    if re.search(r'יד שני[יה]|יד 2|בעלים שני', text):
        return 2
    return None


def _extract_soft_prefs(text: str) -> SoftPreferences:
    flags = {}
    for pref, keywords in _SOFT_KEYWORDS.items():
        flags[pref] = any(kw.lower() in text for kw in keywords)
    return SoftPreferences(**flags)


def _build_semantic_query(raw: str, constraints: HardConstraints) -> str:
    text = raw
    text = re.sub(r'\d+(?:,\d+)?\s*אלף', '', text)
    text = re.sub(r'\d{4,7}', '', text)
    for word in ["עד", "מ", "משנת", "תקציב", "מקסימום", "שנת"]:
        text = re.sub(r'\b' + re.escape(word) + r'\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else raw

# ── Public API ────────────────────────────────────────────────────────────────

def parse_query(query: str) -> ParsedQuery:
    text  = query.strip()
    lower = text.lower()

    make, model = _extract_make_model(lower)
    constraints = HardConstraints(
        make=make,
        model=model,
        price_max=_extract_price_max(lower),
        km_max=_extract_km_max(lower),
        gear_type=_extract_gear_type(lower),
        fuel_type=_extract_fuel_type(lower),
        location=_extract_location(text),
        owners_max=_extract_owners_max(text),
    )
    constraints.year_min, constraints.year_max = _extract_year_range(lower)

    prefs = _extract_soft_prefs(lower)
    if constraints.owners_max == 1:
        prefs.first_owner = True

    has_english = bool(re.search(r'[a-zA-Z]{3,}', text))
    has_hebrew  = bool(re.search(r'[א-ת]', text))
    lang = "en" if has_english and not has_hebrew else "he"

    return ParsedQuery(
        raw_query=text,
        hard_constraints=constraints,
        soft_preferences=prefs,
        semantic_query=_build_semantic_query(text, constraints),
        language=lang,
    )
