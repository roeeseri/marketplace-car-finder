from dataclasses import dataclass
from typing import Optional


@dataclass
class CarFeatures:
    first_owner:       Optional[bool] = None
    single_driver:     Optional[bool] = None
    accident_free:     Optional[bool] = None
    authorized_garage: Optional[bool] = None
    long_test:         Optional[bool] = None
    fuel_efficient:    Optional[bool] = None
    family_car:        Optional[bool] = None
    luxury:            Optional[bool] = None
    off_road:          Optional[bool] = None


@dataclass
class CarAd:
    make: str                            # יצרן
    model: str                           # דגם
    year: int                            # שנת ייצור
    price: float                         # מחיר (₪)
    km: float                            # קילומטראז'
    fuel_type: Optional[str] = None      # סוג דלק
    gear_type: Optional[str] = None      # סוג גיר
    engine_volume: Optional[int] = None  # נפח מנוע (סמ"ק)
    previous_owners: Optional[int] = None
    location: Optional[str] = None       # אזור
    description: Optional[str] = None   # תיאור חופשי
    ad_id: Optional[str] = None
    features: Optional[CarFeatures] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "features"}

    @staticmethod
    def from_dict(data: dict) -> "CarAd":
        return CarAd(
            ad_id=str(data.get("ad_id", "")),
            make=str(data["make"]),
            model=str(data["model"]),
            year=int(data["year"]),
            price=float(data["price"]),
            km=float(data["km"]),
            fuel_type=data.get("fuel_type"),
            gear_type=data.get("gear_type"),
            engine_volume=int(data["engine_volume"]) if data.get("engine_volume") else None,
            previous_owners=int(data["previous_owners"]) if data.get("previous_owners") else None,
            location=data.get("location"),
            description=data.get("description"),
        )
