import os
os.environ.setdefault("PYTHONUTF8", "1")

import traceback
import streamlit as st

from config import FAISS_INDEX_PATH, INDEX_MAP_PATH, SAMPLE_CSV
from src.data.feature_extractor import extract_all
from src.data.loader import load_from_csv
from src.data.preprocessor import preprocess_all
from src.data.validator import validate_all
from src.explanation.explainer import explain
from src.evaluation.report import build_full_report
from src.nlu.query_parser import parse_query
from src.ranking.ranker import RankingWeights, rank
from src.search.embedder import Embedder
from src.search.filter import filter_ads
from src.search.index_builder import build_and_save
from src.search.semantic_search import load_semantic_search

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TOP_N = 10

EXAMPLE_QUERIES = [
    "רכב לסטודנט עד 40 אלף",
    "אוטו קטן לעיר, חסכוני, בלי כאב ראש",
    "טויוטה יד ראשונה ללא תאונות עד 70 אלף",
    "רכב חדש חדש למתחיל עם קילומטראז' נמוך",
    "ב.מ.וו אוטומטי עד 150 אלף",
]

_SOFT_LABELS = {
    "family_car":     "רכב משפחתי",
    "fuel_efficient": "חסכוני בדלק",
    "reliable":       "אמין",
    "first_owner":    "יד ראשונה",
    "young_driver":   "נהג צעיר",
    "off_road":       "שטח / 4×4",
    "luxury":         "יוקרה",
    "city_driving":   "עירוני",
    "long_trips":     "נסיעות ארוכות",
}

_CONSTRAINT_LABELS = [
    ("יצרן",          "make",       None),
    ("דגם",           "model",      None),
    ("מחיר מקסימלי",  "price_max",  lambda v: f"{int(v):,} ₪"),
    ("מחיר מינימלי",  "price_min",  lambda v: f"{int(v):,} ₪"),
    ("שנה מ",         "year_min",   None),
    ("שנה עד",        "year_max",   None),
    ('ק"מ מקסימלי',   "km_max",     lambda v: f"{int(v):,} ק\"מ"),
    ("גיר",           "gear_type",  None),
    ("דלק",           "fuel_type",  None),
    ("מיקום",         "location",   None),
    ("בעלים מקסימלי", "owners_max", None),
]

# ─────────────────────────────────────────────────────────────────────────────
# CSS — full RTL, white theme, modern cards
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
<!-- Google Fonts: Heebo (Hebrew + Latin) -->
<link href="https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;600;700;800&display=swap"
      rel="stylesheet">

<style>
/* ── Global reset ───────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* Apply Heebo font everywhere */
html, body,
.stApp,
[class*="st-"],
.stMarkdown, .stTextInput, .stButton,
button, input, textarea, select {
    font-family: 'Heebo', 'Segoe UI', Arial, sans-serif !important;
}

/* White background for the whole app */
.stApp {
    background-color: #F8FAFC !important;
}

/* Main content: centred, limited width */
.block-container {
    max-width: 860px !important;
    margin: 0 auto !important;
    padding: 1.5rem 2rem 6rem !important;
}

/* ── Hide Streamlit chrome ─────────────────────────────────────────────── */
#MainMenu       { visibility: hidden !important; }
footer          { visibility: hidden !important; }
header          { visibility: hidden !important; }
.stDeployButton { display: none !important; }

/* ── RTL for Streamlit markdown output ────────────────────────────────── */
.stMarkdown p,
.stMarkdown li,
.stMarkdown span {
    direction: rtl;
    text-align: right;
}

/* ── Text input ────────────────────────────────────────────────────────── */
.stTextInput > label { display: none; }

.stTextInput > div > div > input {
    direction: rtl !important;
    text-align: right !important;
    font-size: 1.08rem !important;
    font-weight: 400 !important;
    line-height: 1.5 !important;
    border: 2px solid #CBD5E1 !important;
    border-radius: 14px !important;
    padding: 14px 20px !important;
    background: #FFFFFF !important;
    color: #1E293B !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
    transition: border-color .18s, box-shadow .18s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 4px rgba(37,99,235,0.10) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: #94A3B8 !important; }

/* ── Primary "חפש" button ─────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    padding: 0.65rem 2.4rem !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.28) !important;
    transition: background .18s, box-shadow .18s, transform .12s !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1D4ED8 !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.38) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"]:active { transform: translateY(0) !important; }

/* ── Example query chip buttons ──────────────────────────────────────── */
.stButton > button:not([kind="primary"]) {
    background: #FFFFFF !important;
    color: #2563EB !important;
    border: 1.5px solid #BFDBFE !important;
    border-radius: 999px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.3rem 0.9rem !important;
    transition: background .15s, border-color .15s, color .15s !important;
    white-space: nowrap !important;
    box-shadow: none !important;
}
.stButton > button:not([kind="primary"]):hover {
    background: #EFF6FF !important;
    border-color: #2563EB !important;
    color: #1D4ED8 !important;
}

/* ── Expander ─────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    background: #F8FAFC !important;
    margin-top: 6px !important;
}
[data-testid="stExpander"] summary {
    direction: rtl !important;
    padding: 0.6rem 1rem !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
}
[data-testid="stExpander"] summary:hover { background: #F1F5F9 !important; }
[data-testid="stExpander"] > div:last-child {
    padding: 0.75rem 1rem 1rem !important;
    direction: rtl !important;
}

/* ── Spinner ──────────────────────────────────────────────────────────── */
.stSpinner > div {
    direction: rtl;
    text-align: right;
    color: #2563EB !important;
}

/* ── Alert / warning ──────────────────────────────────────────────────── */
.stAlert {
    direction: rtl !important;
    text-align: right !important;
    border-radius: 10px !important;
}

/* ── Divider ──────────────────────────────────────────────────────────── */
hr { border-color: #E2E8F0 !important; margin: 1.8rem 0 !important; }

/* ══════════════════════════════════════════════════════════════════════
   Custom component styles
══════════════════════════════════════════════════════════════════════ */

/* ── Hero section ─────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #EFF6FF 0%, #F0F9FF 55%, #EDE9FE 100%);
    border: 1px solid #DBEAFE;
    border-radius: 20px;
    padding: 2.8rem 2rem 2.4rem;
    text-align: center;
    margin-bottom: 2rem;
    direction: rtl;
}
.hero-icon  { font-size: 3rem; line-height: 1; margin-bottom: 0.6rem; }
.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #1E3A5F;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}
.hero-sub {
    font-size: 1.0rem;
    font-weight: 400;
    color: #4A6FA5;
    line-height: 1.6;
}

/* ── Search wrapper hint label ────────────────────────────────────────── */
.search-label {
    direction: rtl;
    text-align: right;
    font-size: 0.82rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}

/* ── Example buttons label ────────────────────────────────────────────── */
.examples-label {
    direction: rtl;
    text-align: right;
    font-size: 0.82rem;
    color: #94A3B8;
    margin: 10px 0 6px;
}

/* ── Results header ───────────────────────────────────────────────────── */
.results-header {
    direction: rtl;
    text-align: right;
    font-size: 1.0rem;
    font-weight: 600;
    color: #334155;
    padding: 0.4rem 0 1rem;
    border-bottom: 2px solid #E2E8F0;
    margin-bottom: 1.2rem;
}
.results-header b { color: #2563EB; }
.results-header i { color: #64748B; font-style: normal; font-weight: 400; }

/* ── Car result card ──────────────────────────────────────────────────── */
.car-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1.4rem 1.6rem 1rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 4px 16px rgba(0,0,0,0.04);
    direction: rtl;
    transition: box-shadow .2s, border-color .2s;
}
.car-card:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.07), 0 8px 24px rgba(37,99,235,0.08);
    border-color: #BFDBFE;
}

/* Rank label (top right) */
.rank-label {
    font-size: 0.76rem;
    font-weight: 700;
    color: #94A3B8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    display: block;
    margin-bottom: 2px;
}

/* Car name title */
.car-name {
    font-size: 1.35rem;
    font-weight: 800;
    color: #0F172A;
    letter-spacing: -0.01em;
    line-height: 1.2;
}

/* Score badge — top left (in RTL: visual right) */
.score-badge-total {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #EFF6FF;
    border: 1.5px solid #BFDBFE;
    border-radius: 999px;
    padding: 4px 14px;
    font-size: 0.90rem;
    font-weight: 700;
    color: #1D4ED8;
    white-space: nowrap;
}
.score-badge-total .star { color: #F59E0B; font-size: 1rem; }

/* Price */
.car-price {
    font-size: 1.6rem;
    font-weight: 800;
    color: #2563EB;
    letter-spacing: -0.02em;
    margin: 0.5rem 0 0.4rem;
    line-height: 1;
}

/* Stats row: km, gear, fuel, location */
.stats-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 18px;
    margin: 0.5rem 0 0.7rem;
    direction: rtl;
}
.stat-item {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 0.90rem;
    color: #475569;
    font-weight: 500;
}
.stat-icon { font-size: 0.95rem; }
.stat-val  { font-weight: 600; color: #1E293B; }

/* Feature tags */
.tags-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 0.5rem 0 0.8rem;
    direction: rtl;
}
.feature-tag {
    display: inline-flex;
    align-items: center;
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.80rem;
    font-weight: 600;
    color: #15803D;
}

/* "למה הרכב הזה מתאים?" box */
.why-box {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 12px;
    padding: 0.85rem 1.1rem;
    margin: 0.8rem 0;
    direction: rtl;
}
.why-header {
    font-size: 0.82rem;
    font-weight: 700;
    color: #92400E;
    margin-bottom: 5px;
    letter-spacing: 0.02em;
}
.why-text {
    font-size: 0.95rem;
    font-weight: 400;
    color: #78350F;
    line-height: 1.65;
}

/* Score metric grid — 4 cards */
.score-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    margin-top: 0.9rem;
    direction: rtl;
}
.sc-card {
    border-radius: 10px;
    padding: 0.65rem 0.75rem 0.55rem;
    text-align: center;
    border: 1px solid transparent;
}
.sc-num {
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.sc-lbl {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    margin-top: 4px;
    opacity: 0.75;
}
.sc-bar-wrap {
    background: rgba(0,0,0,0.08);
    border-radius: 3px;
    height: 4px;
    margin: 6px 0 2px;
    overflow: hidden;
}
.sc-bar-fill { height: 4px; border-radius: 3px; }

/* Total score — blue */
.sc-total {
    background: #EFF6FF;
    border-color: #BFDBFE;
    color: #1E40AF;
}
.sc-total .sc-bar-fill { background: #2563EB; }

/* Semantic — indigo */
.sc-semantic {
    background: #EEF2FF;
    border-color: #C7D2FE;
    color: #3730A3;
}
.sc-semantic .sc-bar-fill { background: #6366F1; }

/* Quality — green */
.sc-quality {
    background: #F0FDF4;
    border-color: #BBF7D0;
    color: #166534;
}
.sc-quality .sc-bar-fill { background: #16A34A; }

/* Features — purple */
.sc-features {
    background: #FAF5FF;
    border-color: #E9D5FF;
    color: #6B21A8;
}
.sc-features .sc-bar-fill { background: #9333EA; }

/* ── Technical details (inside expander) ──────────────────────────────── */
.tech-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    direction: rtl;
}
.tech-table tr { border-bottom: 1px solid #F1F5F9; }
.tech-table tr:last-child { border-bottom: none; }
.tech-table td { padding: 5px 6px; vertical-align: top; }
.tech-table td:first-child {
    color: #94A3B8;
    font-weight: 600;
    white-space: nowrap;
    padding-left: 16px;
    width: 40%;
}
.tech-table td:last-child { color: #334155; font-weight: 500; }

.soft-pref-row {
    margin-top: 10px;
    direction: rtl;
    font-size: 0.83rem;
    color: #64748B;
}
.soft-pref-row b { color: #334155; }

.semantic-q {
    margin-top: 10px;
    direction: rtl;
    font-size: 0.82rem;
    color: #64748B;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 6px 10px;
    font-family: monospace;
}

/* ── Empty state ───────────────────────────────────────────────────────── */
.empty-state {
    text-align: center;
    direction: rtl;
    padding: 4rem 2rem;
    color: #94A3B8;
}
.empty-icon  { font-size: 3rem; margin-bottom: 1rem; }
.empty-title { font-size: 1.2rem; font-weight: 700; color: #64748B; margin-bottom: 0.4rem; }
.empty-sub   { font-size: 0.95rem; color: #94A3B8; line-height: 1.6; }
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Cached backend  (unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="טוען מודל שפה…")
def _get_embedder() -> Embedder:
    return Embedder()


@st.cache_resource(show_spinner="בונה אינדקס חיפוש…")
def _get_pipeline():
    embedder = _get_embedder()
    ads = load_from_csv(SAMPLE_CSV)
    ads = preprocess_all(ads)
    valid_ads, invalid = validate_all(ads)
    valid_ads = extract_all(valid_ads)
    build_and_save(valid_ads, embedder)
    searcher = load_semantic_search(FAISS_INDEX_PATH, INDEX_MAP_PATH, embedder)
    return valid_ads, searcher, len(invalid)


@st.cache_data(show_spinner="מריץ הערכות…")
def _get_evaluation_report():
    return build_full_report()


def _is_admin_mode() -> bool:
    if os.getenv("CAR_SEARCH_AGENT_ADMIN") == "1":
        return True
    try:
        params = getattr(st, "query_params", {})
        admin = params.get("admin", "0")
        if isinstance(admin, list):
            admin = admin[0] if admin else "0"
        return str(admin).strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        return False


def run_search(query: str):
    valid_ads, searcher, _ = _get_pipeline()
    parsed = parse_query(query)
    hits = searcher.search(parsed.semantic_query, k=len(valid_ads))
    ad_map = {ad.ad_id: ad for ad in valid_ads}
    candidates = [(ad_map[h.ad_id], h.score) for h in hits if h.ad_id in ad_map]
    candidate_ads = [ad for ad, _ in candidates]
    filtered_ids = {a.ad_id for a in filter_ads(candidate_ads, parsed.hard_constraints)}
    filtered = [(ad, score) for ad, score in candidates if ad.ad_id in filtered_ids]
    if not filtered:
        return parsed, []
    ranked = rank(filtered, parsed, RankingWeights())[:TOP_N]
    pairs = [(r, explain(r.ad, parsed, r)) for r in ranked]
    return parsed, pairs


# ─────────────────────────────────────────────────────────────────────────────
# HTML builders
# ─────────────────────────────────────────────────────────────────────────────

def _score_card(css_class: str, label: str, value: float) -> str:
    pct = int(value * 100)
    return (
        f'<div class="sc-card {css_class}">'
        f'<div class="sc-num">{value:.3f}</div>'
        f'<div class="sc-bar-wrap"><div class="sc-bar-fill" style="width:{pct}%"></div></div>'
        f'<div class="sc-lbl">{label}</div>'
        f'</div>'
    )


def _feature_tag(text: str) -> str:
    return f'<span class="feature-tag">{text}</span>'


def _build_card_html(rank_i: int, result, explanation: str) -> str:
    ad = result.ad
    f = ad.features

    # ── Stats row ────────────────────────────────────────────────────────
    stats = []
    stats.append(f'<div class="stat-item"><span class="stat-icon">🛣️</span>'
                 f'<span class="stat-val">{int(ad.km):,}</span>'
                 f'<span style="color:#94A3B8">ק"מ</span></div>')
    if ad.gear_type:
        stats.append(f'<div class="stat-item"><span class="stat-icon">⚙️</span>'
                     f'<span class="stat-val">{ad.gear_type}</span></div>')
    if ad.fuel_type:
        stats.append(f'<div class="stat-item"><span class="stat-icon">⛽</span>'
                     f'<span class="stat-val">{ad.fuel_type}</span></div>')
    if ad.location:
        stats.append(f'<div class="stat-item"><span class="stat-icon">📍</span>'
                     f'<span class="stat-val">{ad.location}</span></div>')
    if ad.engine_volume:
        stats.append(f'<div class="stat-item"><span class="stat-icon">🔧</span>'
                     f'<span class="stat-val">{ad.engine_volume} סמ"ק</span></div>')
    stats_html = "".join(stats)

    # ── Feature tags ─────────────────────────────────────────────────────
    tags = []
    if f:
        if f.first_owner:       tags.append("✓ יד ראשונה")
        if f.accident_free:     tags.append("✓ ללא תאונות")
        if f.authorized_garage: tags.append("✓ מוסך מורשה")
        if f.long_test:         tags.append("✓ טסט ארוך")
        if f.family_car:        tags.append("✓ משפחתי")
        if f.luxury:            tags.append("✓ יוקרה")
        if f.off_road:          tags.append("✓ שטח")
        if f.fuel_efficient:    tags.append("✓ חסכוני")
    tags_html = "".join(_feature_tag(t) for t in tags)
    tags_section = f'<div class="tags-row">{tags_html}</div>' if tags else ""

    # ── Score grid ────────────────────────────────────────────────────────
    scores_html = (
        '<div class="score-grid">'
        + _score_card("sc-total",    "ציון כולל",      result.total_score)
        + _score_card("sc-semantic", "סמנטי",          result.semantic_score)
        + _score_card("sc-quality",  "איכות רכב",      result.vehicle_quality_score)
        + _score_card("sc-features", "התאמת תכונות",   result.feature_match_score)
        + '</div>'
    )

    return f"""
<div class="car-card">
  <!-- Header row -->
  <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:8px; margin-bottom:4px">
    <div>
      <span class="rank-label">תוצאה #{rank_i}</span>
      <div class="car-name">{ad.make} {ad.model} &nbsp;·&nbsp; {ad.year}</div>
    </div>
    <div class="score-badge-total">
      <span class="star">⭐</span> {result.total_score:.3f}
    </div>
  </div>

  <!-- Price -->
  <div class="car-price">{int(ad.price):,} ₪</div>

  <!-- Stats -->
  <div class="stats-row">{stats_html}</div>

  <!-- Feature tags -->
  {tags_section}

  <!-- Why box -->
  <div class="why-box">
    <div class="why-header">💡 למה הרכב הזה מתאים?</div>
    <div class="why-text">{explanation.replace("הרכב הומלץ מכיוון ש", "").rstrip(".")}</div>
  </div>

  <!-- Score mini-cards -->
  {scores_html}
</div>
"""


def _build_tech_html(result, parsed_query) -> str:
    c = parsed_query.hard_constraints
    p = parsed_query.soft_preferences

    rows = []
    for label, field, fmt in _CONSTRAINT_LABELS:
        val = getattr(c, field, None)
        if val is not None:
            display = fmt(val) if fmt else str(val)
            rows.append(f"<tr><td>{label}</td><td>{display}</td></tr>")

    table_html = (
        f'<table class="tech-table"><tbody>{"".join(rows)}</tbody></table>'
        if rows else
        '<div style="color:#94A3B8;font-size:0.85rem">לא זוהו אילוצים קשים בשאילתה.</div>'
    )

    soft_active = [_SOFT_LABELS.get(k, k) for k, v in p.__dict__.items() if v]
    soft_html = (
        f'<div class="soft-pref-row"><b>העדפות:</b> {", ".join(soft_active)}</div>'
        if soft_active else ""
    )

    sem_q = parsed_query.semantic_query
    sem_html = (
        f'<div style="margin-top:8px;direction:rtl;font-size:0.78rem;color:#94A3B8;font-weight:600">'
        f'שאילתה סמנטית שנשלחה למנוע:</div>'
        f'<div class="semantic-q">{sem_q}</div>'
    )

    lang = "עברית" if parsed_query.language == "he" else "אנגלית"
    lang_html = (
        f'<div style="margin-top:8px;direction:rtl;font-size:0.80rem;color:#94A3B8">'
        f'שפה שזוהתה: <b style="color:#475569">{lang}</b></div>'
    )

    return table_html + soft_html + sem_html + lang_html


# ─────────────────────────────────────────────────────────────────────────────
# Page sections
# ─────────────────────────────────────────────────────────────────────────────

def _render_hero():
    st.markdown("""
    <div class="hero">
      <div class="hero-icon">🚗</div>
      <div class="hero-title">חיפוש רכב חכם</div>
      <div class="hero-sub">מצאו את הרכב המושלם — חפשו בעברית או באנגלית טבעית<br>
      הסוכן מבין דרישות, מדרג תוצאות ומסביר כל המלצה</div>
    </div>
    """, unsafe_allow_html=True)


def _render_search_area():
    st.markdown('<div class="search-label">מה אתם מחפשים?</div>', unsafe_allow_html=True)

    st.text_input(
        label="שאילתת חיפוש",
        placeholder='לדוגמה: "מאזדה 3 אוטומטי עד 70 אלף יד ראשונה"',
        label_visibility="collapsed",
        key="query_input",
    )

    st.markdown('<div class="examples-label">חיפושים לדוגמה:</div>', unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLE_QUERIES))
    for i, ex in enumerate(EXAMPLE_QUERIES):
        if cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["query_input"] = ex
            st.rerun()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    return st.button("🔍 חפש", type="primary")


def _render_results(parsed, pairs, last_query):
    count = len(pairs)
    st.markdown(
        f'<div class="results-header">'
        f'נמצאו <b>{count} תוצאות</b> עבור: <i>"{last_query}"</i>'
        f'</div>',
        unsafe_allow_html=True,
    )

    for i, (result, expl) in enumerate(pairs, 1):
        st.markdown(_build_card_html(i, result, expl), unsafe_allow_html=True)
        with st.expander("🔧 פרטים טכניים", expanded=False):
            st.markdown(_build_tech_html(result, parsed), unsafe_allow_html=True)
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


def _render_empty():
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">🔍</div>
      <div class="empty-title">לא נמצאו תוצאות</div>
      <div class="empty-sub">
        לא מצאנו רכבים התואמים את הדרישות שלכם.<br>
        נסו לשנות את השאילתה, להרחיב את הטווח, או להסיר אילוצים.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="חיפוש רכב חכם",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    # Session state
    for key, default in [("query_input", ""), ("search_results", None), ("search_error", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    _render_hero()
    search_clicked = _render_search_area()

    if _is_admin_mode():
        with st.expander("הערכות לפי פרק 5", expanded=False):
            report = _get_evaluation_report()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("NLU Hard F1", f"{report['nlu']['hard']['f1']:.3f}")
            c2.metric("NLU Soft F1", f"{report['nlu']['soft']['f1']:.3f}")
            c3.metric("NLU Combined F1", f"{report['nlu']['combined']['f1']:.3f}")
            c4.metric("NLU P/R Mean", f"{report['nlu']['combined']['pr_mean']:.3f}")

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "NLU", "Retrieval", "Ablation", "Time / Memory", "Judge"
            ])

            with tab1:
                st.table([{
                    "Metric": "Hard Precision",
                    "Value": f"{report['nlu']['hard']['precision']:.3f}",
                }, {
                    "Metric": "Hard Recall",
                    "Value": f"{report['nlu']['hard']['recall']:.3f}",
                }, {
                    "Metric": "Hard F1",
                    "Value": f"{report['nlu']['hard']['f1']:.3f}",
                }, {
                    "Metric": "Soft Precision",
                    "Value": f"{report['nlu']['soft']['precision']:.3f}",
                }, {
                    "Metric": "Soft Recall",
                    "Value": f"{report['nlu']['soft']['recall']:.3f}",
                }, {
                    "Metric": "Soft F1",
                    "Value": f"{report['nlu']['soft']['f1']:.3f}",
                }, {
                    "Metric": "Combined F1",
                    "Value": f"{report['nlu']['combined']['f1']:.3f}",
                }, {
                    "Metric": "Combined P/R Mean",
                    "Value": f"{report['nlu']['combined']['pr_mean']:.3f}",
                }])

            with tab2:
                st.table([{
                    "Variant": "Smart",
                    "P@5": f"{report['retrieval']['smart'].precision_at_k:.3f}",
                    "R@5": f"{report['retrieval']['smart'].recall_at_k:.3f}",
                    "NDCG@5": f"{report['retrieval']['smart'].ndcg_at_k:.3f}",
                }, {
                    "Variant": "Baseline",
                    "P@5": f"{report['retrieval']['baseline'].precision_at_k:.3f}",
                    "R@5": f"{report['retrieval']['baseline'].recall_at_k:.3f}",
                    "NDCG@5": f"{report['retrieval']['baseline'].ndcg_at_k:.3f}",
                }, {
                    "Variant": "No rerank",
                    "P@5": f"{report['retrieval']['no_rerank'].precision_at_k:.3f}",
                    "R@5": f"{report['retrieval']['no_rerank'].recall_at_k:.3f}",
                    "NDCG@5": f"{report['retrieval']['no_rerank'].ndcg_at_k:.3f}",
                }])
                st.dataframe(report["case_rows"], use_container_width=True)

            with tab3:
                st.table([{
                    "Variant": "No semantic",
                    "P@5": f"{report['ablation']['no_semantic'].precision_at_k:.3f}",
                    "R@5": f"{report['ablation']['no_semantic'].recall_at_k:.3f}",
                    "NDCG@5": f"{report['ablation']['no_semantic'].ndcg_at_k:.3f}",
                }, {
                    "Variant": "No rerank",
                    "P@5": f"{report['ablation']['no_rerank'].precision_at_k:.3f}",
                    "R@5": f"{report['ablation']['no_rerank'].recall_at_k:.3f}",
                    "NDCG@5": f"{report['ablation']['no_rerank'].ndcg_at_k:.3f}",
                }])

            with tab4:
                st.table([{
                    "Variant": "Smart",
                    "Mean ms": f"{report['timings']['smart'].mean_ms:.2f}",
                    "Median ms": f"{report['timings']['smart'].median_ms:.2f}",
                    "Peak KB": f"{report['timings']['smart'].peak_kb:.1f}",
                }, {
                    "Variant": "Baseline",
                    "Mean ms": f"{report['timings']['baseline'].mean_ms:.2f}",
                    "Median ms": f"{report['timings']['baseline'].median_ms:.2f}",
                    "Peak KB": f"{report['timings']['baseline'].peak_kb:.1f}",
                }])

            with tab5:
                st.write(report["judge_sample"][:20])

    # Run search
    if search_clicked:
        q = st.session_state["query_input"].strip()
        if not q:
            st.warning("אנא הכניסו שאילתת חיפוש.")
        else:
            with st.spinner("מחפש…"):
                try:
                    parsed, pairs = run_search(q)
                    st.session_state["search_results"] = (parsed, pairs, q)
                    st.session_state["search_error"] = None
                except Exception:
                    st.session_state["search_results"] = None
                    st.session_state["search_error"] = traceback.format_exc()

    # Error
    if st.session_state["search_error"]:
        st.error("אירעה שגיאה בעת החיפוש.")
        with st.expander("פרטי שגיאה"):
            st.code(st.session_state["search_error"])
        return

    # No search yet
    if st.session_state["search_results"] is None:
        return

    # Results
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    parsed, pairs, last_query = st.session_state["search_results"]

    if not pairs:
        _render_empty()
    else:
        _render_results(parsed, pairs, last_query)


if __name__ == "__main__":
    main()
