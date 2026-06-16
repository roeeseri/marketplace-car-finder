from typing import List

from src.data.schema import CarAd
from src.location_groups import location_matches
from src.nlu.query_parser import HardConstraints


def filter_ads(ads: List[CarAd], constraints: HardConstraints) -> List[CarAd]:
    return [ad for ad in ads if _matches(ad, constraints)]


def _matches(ad: CarAd, c: HardConstraints) -> bool:
    if c.make and ad.make and c.make.lower() != ad.make.lower():
        return False
    if c.model and ad.model and c.model.lower() != ad.model.lower():
        return False
    if c.price_max is not None and ad.price > c.price_max:
        return False
    if c.price_min is not None and ad.price < c.price_min:
        return False
    if c.year_min is not None and ad.year < c.year_min:
        return False
    if c.year_max is not None and ad.year > c.year_max:
        return False
    if c.km_max is not None and ad.km > c.km_max:
        return False
    if c.gear_type and ad.gear_type and c.gear_type.lower() != ad.gear_type.lower():
        return False
    if c.fuel_type and ad.fuel_type and c.fuel_type.lower() != ad.fuel_type.lower():
        return False
    if c.location:
        if not ad.location or not location_matches(c.location, ad.location):
            return False
    if c.owners_max is not None and ad.previous_owners is not None:
        if ad.previous_owners > c.owners_max:
            return False
    return True
