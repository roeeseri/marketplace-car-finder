import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from config import FAISS_INDEX_PATH, INDEX_MAP_PATH, SAMPLE_CSV
from src.data.feature_extractor import extract_all
from src.data.loader import load_from_csv
from src.data.preprocessor import preprocess_all
from src.data.validator import validate_all
from src.explanation.explainer import explain
from src.nlu.query_parser import parse_query
from src.ranking.ranker import RankingWeights, rank
from src.search.embedder import Embedder
from src.search.filter import filter_ads
from src.search.index_builder import build_and_save
from src.search.semantic_search import load_semantic_search

TOP_N = 5


def run(query: str, top_n: int = TOP_N) -> None:
    print(f"\n{'='*60}")
    print(f"שאילתה: {query}")
    print('='*60)

    # 1. Load & prepare ads
    ads = load_from_csv(SAMPLE_CSV)
    ads = preprocess_all(ads)
    valid_ads, invalid = validate_all(ads)
    if invalid:
        print(f"[warning] {len(invalid)} ads failed validation and were skipped.")

    # 2. Extract features from descriptions
    valid_ads = extract_all(valid_ads)

    # 3. Build FAISS index (rebuild each run for demo; in production build once)
    print("Building index...")
    embedder = Embedder()
    build_and_save(valid_ads, embedder)

    # 4. Parse query
    parsed = parse_query(query)
    print(f"Parsed make={parsed.hard_constraints.make}  "
          f"model={parsed.hard_constraints.model}  "
          f"price_max={parsed.hard_constraints.price_max}  "
          f"gear={parsed.hard_constraints.gear_type}")

    # 5. Semantic search
    searcher  = load_semantic_search(FAISS_INDEX_PATH, INDEX_MAP_PATH, embedder)
    hits      = searcher.search(parsed.semantic_query, k=min(len(valid_ads), 50))
    ad_map    = {ad.ad_id: ad for ad in valid_ads}
    candidates = [(ad_map[h.ad_id], h.score) for h in hits if h.ad_id in ad_map]

    # 6. Hard filter
    candidate_ads    = [ad for ad, _ in candidates]
    filtered_ads     = set(a.ad_id for a in filter_ads(candidate_ads, parsed.hard_constraints))
    filtered         = [(ad, score) for ad, score in candidates if ad.ad_id in filtered_ads]

    if not filtered:
        print("\nלא נמצאו תוצאות התואמות את הדרישות.")
        return

    # 7. Rank
    ranked = rank(filtered, parsed, RankingWeights())

    # 8. Print results
    print(f"\nנמצאו {len(ranked)} תוצאות. מציג את {min(top_n, len(ranked))} הראשונות:\n")
    for i, result in enumerate(ranked[:top_n], 1):
        ad = result.ad
        print(f"  #{i}  {ad.make} {ad.model} {ad.year}  |  "
              f"{int(ad.price):,} ₪  |  {int(ad.km):,} ק\"מ  |  "
              f"{ad.gear_type or ''}  |  {ad.location or ''}")
        print(f"       ציון: {result.total_score:.3f}  "
              f"(סמנטי={result.semantic_score:.3f}, "
              f"איכות={result.vehicle_quality_score:.3f}, "
              f"תכונות={result.feature_match_score:.3f})")
        print(f"       {explain(ad, parsed, result)}")
        print()


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "מאזדה 3 אוטומטי עד 70 אלף"
    run(query)
