from pathlib import Path

BASE_DIR = Path(__file__).parent

# Data paths
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
SAMPLE_CSV = DATA_RAW_DIR / "used_cars_505_ads_with_upgrade_text.csv"
SMALL_SAMPLE_CSV = DATA_RAW_DIR / "sample_ads.csv"
PROCESSED_CSV = DATA_PROCESSED_DIR / "ads_processed.csv"
EMBEDDINGS_PATH = DATA_PROCESSED_DIR / "embeddings.npy"

# Schema field names
FIELD_AD_ID = "ad_id"
FIELD_MAKE = "make"
FIELD_MODEL = "model"
FIELD_YEAR = "year"
FIELD_PRICE = "price"
FIELD_KM = "km"
FIELD_FUEL_TYPE = "fuel_type"
FIELD_GEAR_TYPE = "gear_type"
FIELD_ENGINE_VOLUME = "engine_volume"
FIELD_PREVIOUS_OWNERS = "previous_owners"
FIELD_LOCATION = "location"
FIELD_DESCRIPTION = "description"

REQUIRED_FIELDS = [FIELD_MAKE, FIELD_MODEL, FIELD_YEAR, FIELD_PRICE, FIELD_KM]

# Valid value sets
VALID_FUEL_TYPES = {"בנזין", "דיזל", "היברידי", "חשמלי", "גז"}
VALID_GEAR_TYPES = {"ידני", "אוטומטי"}

# Reasonable range constraints
MIN_YEAR = 1990
MAX_YEAR = 2026
MIN_PRICE = 1000
MAX_PRICE = 2_000_000
MIN_KM = 0
MAX_KM = 1_000_000

# Embeddings and FAISS
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
FAISS_INDEX_PATH = DATA_PROCESSED_DIR / "faiss_index.bin"
INDEX_MAP_PATH = DATA_PROCESSED_DIR / "index_map.json"
TOP_K = 100
EMBEDDING_BATCH_SIZE = 64
