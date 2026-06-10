import sqlite3
from pathlib import Path
from typing import List

import pandas as pd

from src.data.schema import CarAd


def load_from_csv(path: str | Path) -> List[CarAd]:
    df = pd.read_csv(path, dtype=str)
    return [CarAd.from_dict(row) for row in df.to_dict(orient="records")]


def load_from_sqlite(db_path: str | Path, table: str = "ads") -> List[CarAd]:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    finally:
        conn.close()
    return [CarAd.from_dict(row) for row in df.to_dict(orient="records")]


def load_to_dataframe(path: str | Path) -> pd.DataFrame:
    """Load CSV and return a raw DataFrame (useful for preprocessing steps)."""
    return pd.read_csv(path)
