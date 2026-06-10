from typing import List

from src.data.schema import CarAd, CarFeatures

_KEYWORDS = {
    "first_owner":       ["יד ראשונה", "יד 1", "בעלים ראשון", "בעלים ראשונה", "יד-ראשונה"],
    "single_driver":     ["נהג יחיד", "נהגת יחידה", "נהג אחד"],
    "accident_free":     ["ללא תאונות", "ללא נזקים", "נקי מתאונות", "לא היה בתאונה", "ללא היסטוריית תאונות"],
    "authorized_garage": ["מוסך מורשה", "שירות מורשה", "טיפולים מורשה", "אחזקה מורשה"],
    "long_test":         ["טסט ארוך", "טסט עד 20", "טסט עד 202"],
    "fuel_efficient":    ["חסכוני בדלק", "חסכוני", "צריכה נמוכה", "חסכון בדלק", "חסכנות בדלק"],
    "family_car":        ["מתאים למשפחה", "רכב משפחתי", "מרווח", "למשפחה", "7 מושבים"],
    "luxury":            ["יוקרה", "פרימיום", "לוקסוס", "ליין עליון", "חבילת יוקרה", "יוקרתי"],
    "off_road":          ["שטח", "4x4", "4X4", "כונן כפול", "ארבע על ארבע"],
}


def extract_features(ad: CarAd) -> CarFeatures:
    text = (ad.description or "").lower()
    kwargs = {}
    for feature, keywords in _KEYWORDS.items():
        matched = any(kw.lower() in text for kw in keywords)
        kwargs[feature] = True if matched else None
    return CarFeatures(**kwargs)


def extract_all(ads: List[CarAd]) -> List[CarAd]:
    for ad in ads:
        ad.features = extract_features(ad)
    return ads
