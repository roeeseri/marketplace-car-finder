import sqlite3
from pathlib import Path
from typing import List

import pandas as pd

from src.data.schema import CarAd


def save_to_csv(ads: List[CarAd], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([ad.to_dict() for ad in ads]).to_csv(path, index=False, encoding="utf-8-sig")


def save_to_sqlite(ads: List[CarAd], db_path: str | Path, table: str = "ads") -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([ad.to_dict() for ad in ads])
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table, conn, if_exists="replace", index=False)
    finally:
        conn.close()
