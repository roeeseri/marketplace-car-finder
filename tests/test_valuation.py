from src.data.schema import CarAd
from src.valuation.calculator import estimate_car_value


def test_estimate_uses_similar_ads():
    ads = [
        CarAd(make="יונדאי", model="i20", year=2024, price=62000, km=12000, gear_type="אוטומטי", location="חיפה"),
        CarAd(make="יונדאי", model="i20", year=2023, price=58000, km=18000, gear_type="אוטומטי", location="תל אביב"),
        CarAd(make="יונדאי", model="i20", year=2022, price=53000, km=25000, gear_type="אוטומטי", location="רעננה"),
        CarAd(make="יונדאי", model="i20", year=2021, price=47000, km=38000, gear_type="אוטומטי", location="חיפה"),
    ]
    target = CarAd(make="יונדאי", model="i20", year=2024, price=0, km=10000, gear_type="אוטומטי", location="חיפה")
    result = estimate_car_value(target, ads)
    assert result.comparable_count >= 3
    assert result.estimated_price > 0
    assert result.low_price <= result.estimated_price <= result.high_price
