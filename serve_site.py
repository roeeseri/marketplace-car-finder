from __future__ import annotations

import html
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config import SAMPLE_CSV
from src.data.feature_extractor import extract_all
from src.data.loader import load_from_csv
from src.data.preprocessor import preprocess_all
from src.data.validator import validate_all
from src.data.schema import CarAd
from src.explanation.explainer import explain
from src.knowledge.vehicle_insights import build_insight_bundle
from src.nlu.query_parser import parse_query
from src.ranking.ranker import RankingWeights, rank
from src.valuation.calculator import estimate_car_value
from src.search.embedder import Embedder
from src.search.filter import filter_ads
from src.search.index_builder import build_faiss_index, build_id_map
from src.search.semantic_search import SemanticSearch


PORT = 8501


def _load_ads():
    ads = load_from_csv(SAMPLE_CSV)
    ads = preprocess_all(ads)
    valid_ads, _ = validate_all(ads)
    return extract_all(valid_ads)


ADS = _load_ads()
EMBEDDER = Embedder()
SEARCH_INDEX = build_faiss_index(EMBEDDER.encode_ads(ADS))
SEARCHER = SemanticSearch(SEARCH_INDEX, build_id_map(ADS), EMBEDDER)
AD_MAP = {str(ad.ad_id): ad for ad in ADS}


def _smart_search(query: str, top_n: int = 10):
    parsed = parse_query(query)
    hits = SEARCHER.search(parsed.semantic_query, k=len(ADS))
    candidates = [(AD_MAP[h.ad_id], h.score) for h in hits if h.ad_id in AD_MAP]
    allowed_ids = {a.ad_id for a in filter_ads([ad for ad, _ in candidates], parsed.hard_constraints)}
    filtered = [pair for pair in candidates if pair[0].ad_id in allowed_ids]
    if not filtered:
        return parsed, []
    ranked = rank(filtered, parsed, RankingWeights())[:top_n]
    return parsed, [
        (result, explain(result.ad, parsed, result), build_insight_bundle(result.ad, parsed, result))
        for result in ranked
    ]


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _render_valuation_block(params: dict) -> str:
    calc_make = params.get("calc_make", [""])[0]
    calc_model = params.get("calc_model", [""])[0]
    calc_year = _parse_int(params.get("calc_year", [""])[0])
    calc_km = _parse_float(params.get("calc_km", [""])[0])
    calc_gear = params.get("calc_gear", [""])[0]
    calc_fuel = params.get("calc_fuel", [""])[0]
    calc_owners = _parse_int(params.get("calc_owners", [""])[0])
    calc_location = params.get("calc_location", [""])[0]
    calc_condition = params.get("calc_condition", [""])[0]
    valuation_html = ""

    if calc_make and calc_year is not None and calc_km is not None:
        target = CarAd(
            ad_id="calc",
            make=calc_make,
            model=calc_model or "",
            year=calc_year,
            price=0,
            km=calc_km,
            gear_type=calc_gear or None,
            fuel_type=calc_fuel or None,
            previous_owners=calc_owners,
            location=calc_location or None,
            description="",
        )
        estimate = estimate_car_value(
            target=target,
            ads=ADS,
            target_condition=calc_condition or None,
        )
        max_count = max(estimate.histogram_counts) if estimate.histogram_counts else 1
        histogram_html = "".join(
            f"""
            <div class="hist-row">
              <div class="hist-label">{html.escape(label)}</div>
              <div class="hist-track"><div class="hist-fill" style="width:{max(8, int((count / max_count) * 100))}%"></div></div>
              <div class="hist-count">{count}</div>
            </div>
            """
            for label, count in zip(estimate.histogram_labels, estimate.histogram_counts)
        )
        valuation_html = f"""
        <div class="valuation-result">
          <div class="valuation-head">
            <div>
              <div class="valuation-kicker">הערכת שווי משוערת</div>
              <h3>{html.escape(calc_make)} {html.escape(calc_model or '')}</h3>
            </div>
            <div class="valuation-score">אמון {estimate.confidence:.2f}</div>
          </div>
          <div class="valuation-range">
            {estimate.low_price:,} &#8362; - {estimate.high_price:,} &#8362;
          </div>
          <div class="valuation-price">{estimate.estimated_price:,} &#8362;</div>
          <div class="valuation-note">מבוסס על {estimate.comparable_count} רכבים דומים מתוך הדאטה שלנו.</div>
          <div class="valuation-reasons">
            {"".join(f'<span class="valuation-chip">{html.escape(reason)}</span>' for reason in estimate.reasons)}
          </div>
          <div class="valuation-preview">
            {"".join(f'<div class="valuation-preview-item">{html.escape(item)}</div>' for item in estimate.comparable_preview)}
          </div>
          <div class="valuation-hist">
            <div class="mini-title">התפלגות מחירי רכבים דומים</div>
            <div class="histogram">{histogram_html}</div>
          </div>
        </div>
        """

    return f"""
    <section class="section" id="valuation">
      <div class="section-head">
        <h2>מחשבון שווי</h2>
        <p>הערכה מהירה על בסיס רכבים דומים מהדאטה שלנו, כדי להבין אם הרכב יקר או זול יחסית לשוק.</p>
      </div>
      <form class="valuation-form" id="valuation-form" method="post" action="/#valuation">
        <input type="hidden" name="q" value="{html.escape(params.get('q', [''])[0])}" />
        <div class="valuation-grid">
          <input type="text" id="calc_make" name="calc_make" placeholder="יצרן" value="{html.escape(calc_make)}" />
          <input type="text" id="calc_model" name="calc_model" placeholder="דגם" value="{html.escape(calc_model)}" />
          <input type="number" id="calc_year" name="calc_year" placeholder="שנתון" value="{'' if calc_year is None else calc_year}" />
          <input type="number" id="calc_km" name="calc_km" placeholder="ק״מ" value="{'' if calc_km is None else int(calc_km)}" />
          <input type="text" id="calc_gear" name="calc_gear" placeholder="תיבה" value="{html.escape(calc_gear)}" />
          <input type="text" id="calc_fuel" name="calc_fuel" placeholder="דלק" value="{html.escape(calc_fuel)}" />
          <input type="number" id="calc_owners" name="calc_owners" placeholder="מספר בעלים" value="{'' if calc_owners is None else calc_owners}" />
          <input type="text" id="calc_location" name="calc_location" placeholder="מיקום" value="{html.escape(calc_location)}" />
          <select id="calc_condition" name="calc_condition">
            <option value="">מצב הרכב</option>
            <option value="מעולה" {"selected" if calc_condition == "מעולה" else ""}>מעולה</option>
            <option value="טוב" {"selected" if calc_condition == "טוב" else ""}>טוב</option>
            <option value="סביר" {"selected" if calc_condition == "סביר" else ""}>סביר</option>
            <option value="דורש השקעה" {"selected" if calc_condition == "דורש השקעה" else ""}>דורש השקעה</option>
          </select>
        </div>
        <button type="submit">חשב שווי</button>
      </form>
      <div class="valuation-help">אפשר גם ללחוץ על "חשב שווי" מתוך כל כרטיס תוצאה כדי למלא את הטופס אוטומטית.</div>
      {valuation_html}
    </section>
    """


def _render_page(query: str = "", results=None, params: dict | None = None) -> str:
    params = params or {}
    result_cards = ""
    if results:
        for i, (result, explanation, insight) in enumerate(results, 1):
            ad = result.ad
            similar_html = "".join(
                f'<span class="chip">{html.escape(text)}</span>'
                for text in insight.similarity_reasons
            )
            source = insight.source
            source_block = ""
            if source:
                source_text = source.excerpt or source.summary or ""
                source_block = f"""
                <details class="panel-mini">
                  <summary class="mini-title">מקור רשמי לדגם</summary>
                  <a class="source-link" href="{html.escape(source.source_url)}" target="_blank" rel="noreferrer">
                    {html.escape(source.source_name)}
                  </a>
                  <div class="source-body">{html.escape(source_text)}</div>
                </details>
                """
            result_cards += f"""
            <article class="card">
              <div class="card-top">
                <div>
                  <div class="rank">תוצאה #{i}</div>
                  <h3>{html.escape(ad.make)} {html.escape(ad.model)} {ad.year}</h3>
                </div>
                <div class="score">{result.total_score:.3f}</div>
              </div>
              <div class="meta">{int(ad.price):,} &#8362; &middot; {int(ad.km):,} ק"מ &middot; {html.escape(ad.gear_type or '')} &middot; {html.escape(ad.location or '')}</div>
              <div class="explanation">{html.escape(explanation)}</div>
              <div class="insight">
                <div class="panel-mini">
                  <div class="mini-title">למה הרכב דומה?</div>
                  <div class="chips">{similar_html}</div>
                </div>
                {source_block}
              </div>
              <div class="card-actions">
                <button
                  type="button"
                  class="calc-link"
                  data-calc-make="{html.escape(ad.make)}"
                  data-calc-model="{html.escape(ad.model)}"
                  data-calc-year="{ad.year}"
                  data-calc-km="{int(ad.km)}"
                  data-calc-gear="{html.escape(ad.gear_type or '')}"
                  data-calc-fuel="{html.escape(ad.fuel_type or '')}"
                  data-calc-owners="{html.escape(str(ad.previous_owners or ''))}"
                  data-calc-location="{html.escape(ad.location or '')}"
                >חשב שווי במחשבון</button>
              </div>
            </article>
            """
    else:
        result_cards = """
        <div class="empty-state">
          כתוב חיפוש, למשל: רכב קטן לעיר, חסכוני, אוטומטי
        </div>
        """

    return f"""<!doctype html>
<html lang="he" dir="rtl">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>מנוע חיפוש רכבים</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Heebo:wght@400;500;700;800;900&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #f4f7fb;
      --panel: rgba(255,255,255,.92);
      --card: #ffffff;
      --text: #0f172a;
      --muted: #516074;
      --line: #dbe4f0;
      --accent: #0f766e;
      --accent-2: #155e75;
      --accent-soft: #ecfeff;
      --shadow: 0 16px 44px rgba(15, 23, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Heebo", "Segoe UI", Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top right, rgba(15,118,110,.12), transparent 26%),
        radial-gradient(circle at top left, rgba(21,94,117,.10), transparent 24%),
        linear-gradient(180deg, #ffffff 0%, var(--bg) 38%, #eef4f8 100%);
    }}
    .wrap {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 30px 18px 54px;
    }}
    .hero, .card, .section, .empty-state {{
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    .hero {{
      border-radius: 28px;
      padding: 26px;
      overflow: hidden;
      position: relative;
      margin-bottom: 18px;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      inset: auto -80px -100px auto;
      width: 240px;
      height: 240px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(15,118,110,.16), transparent 70%);
      pointer-events: none;
    }}
    .eyebrow {{
      color: var(--accent);
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: .08em;
      font-size: 12px;
      margin-bottom: 10px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(30px, 4vw, 46px);
      line-height: 1.04;
    }}
    .lede {{
      margin: 12px 0 0;
      color: var(--muted);
      max-width: 760px;
      font-size: 17px;
      line-height: 1.7;
    }}
    .search {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 18px;
    }}
    .search input {{
      flex: 1;
      min-width: 280px;
      border: 1px solid #cdd8e6;
      border-radius: 18px;
      padding: 15px 16px;
      font-size: 16px;
      outline: none;
      background: white;
    }}
    .search button {{
      border: 0;
      border-radius: 18px;
      padding: 15px 18px;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: white;
      font-weight: 800;
      cursor: pointer;
      box-shadow: 0 10px 24px rgba(15,118,110,.22);
    }}
    .hint {{
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .section {{
      border-radius: 24px;
      padding: 22px;
    }}
    .section-head {{
      margin-bottom: 14px;
    }}
    .section-head h2 {{
      margin: 0;
      font-size: 22px;
    }}
    .section-head p {{
      margin: 6px 0 0;
      color: var(--muted);
    }}
    .results {{
      display: grid;
      gap: 14px;
    }}
    .card {{
      border-radius: 22px;
      padding: 18px;
    }}
    .card-top {{
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 12px;
    }}
    .rank {{
      color: var(--accent);
      font-size: 12px;
      font-weight: 900;
      letter-spacing: .08em;
      text-transform: uppercase;
    }}
    h3 {{
      margin: 6px 0 0;
      font-size: 24px;
    }}
    .score {{
      background: var(--accent-soft);
      color: var(--accent);
      border: 1px solid #bff3ef;
      border-radius: 999px;
      padding: 8px 14px;
      font-weight: 900;
      white-space: nowrap;
    }}
    .meta {{
      margin-top: 10px;
      font-weight: 700;
      color: #334155;
    }}
    .explanation {{
      margin-top: 12px;
      padding: 12px 14px;
      border-radius: 16px;
      border: 1px solid #dcecf1;
      background: #f7fbfd;
      color: #334155;
      line-height: 1.7;
    }}
    .insight {{
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }}
    .card-actions {{
      margin-top: 14px;
    }}
    .calc-link {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 11px 14px;
      border-radius: 14px;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: white;
      font: inherit;
      font-weight: 800;
      border: 0;
      cursor: pointer;
      text-decoration: none;
      box-shadow: 0 10px 24px rgba(15,118,110,.18);
    }}
    .panel-mini {{
      border-radius: 18px;
      padding: 14px;
      border: 1px solid #dcecf1;
      background: linear-gradient(180deg, #fff 0%, #f8fbfd 100%);
    }}
    details.panel-mini {{
      cursor: default;
    }}
    details.panel-mini > summary {{
      list-style: none;
    }}
    details.panel-mini > summary::-webkit-details-marker {{
      display: none;
    }}
    .mini-title {{
      font-size: 12px;
      font-weight: 900;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: .08em;
      margin-bottom: 10px;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 7px 10px;
      background: #eefbf9;
      color: #115e59;
      border: 1px solid #c6f0e8;
      font-size: 12px;
      font-weight: 700;
    }}
    .source-link {{
      display: inline-block;
      color: #0f172a;
      text-decoration: none;
      font-weight: 900;
    }}
    .source-link:hover {{
      text-decoration: underline;
    }}
    .source-body {{
      margin-top: 8px;
      color: var(--muted);
      line-height: 1.65;
      font-size: 13px;
    }}
    .empty-state {{
      border-radius: 18px;
      padding: 18px;
      color: var(--muted);
    }}
    .valuation-form {{
      display: grid;
      gap: 12px;
    }}
      .valuation-grid {{
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
      }}
      .valuation-grid input {{
        border: 1px solid #cdd8e6;
        border-radius: 14px;
        padding: 12px 14px;
        font: inherit;
        background: white;
      }}
      .valuation-grid select {{
        border: 1px solid #cdd8e6;
        border-radius: 14px;
        padding: 12px 14px;
        font: inherit;
        background: white;
      }}
    .valuation-form button {{
      width: fit-content;
      border: 0;
      border-radius: 14px;
      padding: 12px 18px;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      color: white;
      font-weight: 800;
      cursor: pointer;
    }}
    .valuation-help {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .valuation-result {{
      margin-top: 16px;
      border-radius: 20px;
      padding: 18px;
      background: linear-gradient(180deg, #ffffff 0%, #f7fbfd 100%);
      border: 1px solid #dcecf1;
    }}
    .valuation-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
    }}
    .valuation-kicker {{
      font-size: 12px;
      font-weight: 900;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: .08em;
    }}
    .valuation-score {{
      border-radius: 999px;
      border: 1px solid #bfebdf;
      background: #ecfdf5;
      color: #166534;
      font-weight: 900;
      padding: 8px 12px;
    }}
    .valuation-range {{
      margin-top: 10px;
      color: #0f766e;
      font-size: 15px;
      font-weight: 800;
    }}
    .valuation-price {{
      margin-top: 4px;
      font-size: clamp(26px, 3.8vw, 38px);
      font-weight: 900;
      letter-spacing: -0.04em;
    }}
    .valuation-note {{
      margin-top: 6px;
      color: var(--muted);
    }}
    .valuation-reasons {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 14px;
    }}
    .valuation-chip {{
      border-radius: 999px;
      background: #eefbf9;
      border: 1px solid #c6f0e8;
      color: #115e59;
      padding: 7px 10px;
      font-size: 12px;
      font-weight: 700;
    }}
    .valuation-preview {{
      margin-top: 14px;
      display: grid;
      gap: 8px;
    }}
    .valuation-preview-item {{
      padding: 10px 12px;
      border-radius: 14px;
      border: 1px solid #e2e8f0;
      background: white;
      color: #334155;
    }}
    .valuation-hist {{
      margin-top: 16px;
    }}
    .histogram {{
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }}
    .hist-row {{
      display: grid;
      grid-template-columns: minmax(110px, 180px) 1fr auto;
      gap: 10px;
      align-items: center;
    }}
    .hist-label {{
      font-size: 12px;
      color: var(--muted);
    }}
    .hist-track {{
      position: relative;
      height: 12px;
      border-radius: 999px;
      background: #e2e8f0;
      overflow: hidden;
    }}
    .hist-fill {{
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
    }}
    .hist-count {{
      font-size: 12px;
      font-weight: 800;
      color: #0f172a;
    }}
    @media (max-width: 640px) {{
      .wrap {{ padding: 16px 12px 40px; }}
      .hero, .section, .card {{ border-radius: 20px; }}
      .card-top {{ flex-direction: column; }}
      h3 {{ font-size: 22px; }}
      .valuation-grid {{ grid-template-columns: 1fr 1fr; }}
      .hist-row {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">Car Search Agent</div>
      <h1>חיפוש רכבים בעברית, עם הסבר ומקור רשמי לכל תוצאה</h1>
      <p class="lede">הקלד שאילתה חופשית, לדוגמה: רכב קטן לעיר, חסכוני, אוטומטי. המערכת תציג תוצאות עם הסבר למה הן מתאימות, למה הן דומות, וקישור לעמוד הרשמי של הדגם כשיש לנו אותו.</p>
      <form class="search" method="get" action="/">
        <input type="hidden" name="page" value="search" />
        <input type="text" name="q" value="{html.escape(query)}" placeholder="למשל: רכב טוב בחיפה" />
        <button type="submit">חפש</button>
      </form>
      <div class="hint">אין כאן דוגמאות מוכנות. כל חיפוש נבנה מתוך השאילתה שלך בלבד.</div>
    </section>

    <section class="section">
      <div class="section-head">
        <h2>תוצאות</h2>
        <p>{html.escape(query) if query else 'כתוב שאילתה כדי לראות תוצאות.'}</p>
      </div>
      <div class="results">{result_cards}</div>
    </section>

    {_render_valuation_block(params)}
  </div>
  <script>
    (() => {{
        const valuationSection = document.getElementById('valuation');
        const valuationForm = document.getElementById('valuation-form');
      const fields = {{
        make: document.getElementById('calc_make'),
        model: document.getElementById('calc_model'),
        year: document.getElementById('calc_year'),
        km: document.getElementById('calc_km'),
        gear: document.getElementById('calc_gear'),
        fuel: document.getElementById('calc_fuel'),
        owners: document.getElementById('calc_owners'),
        location: document.getElementById('calc_location'),
        condition: document.getElementById('calc_condition'),
      }};
      document.querySelectorAll('.calc-link').forEach((button) => {{
        button.addEventListener('click', () => {{
          if (fields.make) fields.make.value = button.dataset.calcMake || '';
          if (fields.model) fields.model.value = button.dataset.calcModel || '';
          if (fields.year) fields.year.value = button.dataset.calcYear || '';
          if (fields.km) fields.km.value = button.dataset.calcKm || '';
          if (fields.gear) fields.gear.value = button.dataset.calcGear || '';
          if (fields.fuel) fields.fuel.value = button.dataset.calcFuel || '';
          if (fields.owners) fields.owners.value = button.dataset.calcOwners || '';
          if (fields.location) fields.location.value = button.dataset.calcLocation || '';
          if (fields.condition) fields.condition.value = '';
          if (valuationSection) {{
            valuationSection.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
          }}
          if (valuationForm && typeof valuationForm.requestSubmit === 'function') {{
            valuationForm.requestSubmit();
          }} else if (valuationForm) {{
            valuationForm.submit();
          }}
        }});
      }});
    }})();
  </script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        query = params.get("q", [""])[0].strip()
        results = []
        if query:
            _, results = _smart_search(query)
        body = _render_page(query, results, params).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0") or "0")
            raw_body = self.rfile.read(content_length).decode("utf-8", errors="replace")
            params = urllib.parse.parse_qs(raw_body)
            query = params.get("q", [""])[0].strip()
            results = []
            if query:
                _, results = _smart_search(query)
            body = _render_page(query, results, params).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            error_body = f"<pre>POST failed: {html.escape(repr(exc))}</pre>".encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(error_body)))
            self.end_headers()
            self.wfile.write(error_body)

    def log_message(self, format, *args):
        return


def main():
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Serving on http://127.0.0.1:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
