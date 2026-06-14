from __future__ import annotations

import html
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config import SAMPLE_CSV
from src.data.feature_extractor import extract_all
from src.data.loader import load_from_csv
from src.data.preprocessor import preprocess_all
from src.data.validator import validate_all
from src.evaluation.report import build_full_report
from src.explanation.explainer import explain
from src.nlu.query_parser import parse_query
from src.ranking.ranker import RankingWeights, rank
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
REPORT = None

EXAMPLE_QUERIES = [
    "Mazda 3 automatic up to 70k",
    "Toyota first owner without accidents",
    "Hybrid up to 100k fuel efficient",
    "BMW automatic up to 150k",
    "Family car with low mileage",
]


def _get_report():
    global REPORT
    if REPORT is None:
        REPORT = build_full_report()
    return REPORT


def _smart_search(query: str, top_n: int = 10):
    parsed = parse_query(query)
    hits = SEARCHER.search(parsed.semantic_query, k=len(ADS))
    candidates = [(AD_MAP[h.ad_id], h.score) for h in hits if h.ad_id in AD_MAP]
    allowed_ids = {a.ad_id for a in filter_ads([ad for ad, _ in candidates], parsed.hard_constraints)}
    filtered = [pair for pair in candidates if pair[0].ad_id in allowed_ids]
    if not filtered:
        return parsed, []
    ranked = rank(filtered, parsed, RankingWeights())[:top_n]
    pairs = [(result, explain(result.ad, parsed, result)) for result in ranked]
    return parsed, pairs


def _nav(active: str) -> str:
    items = [
        ("overview", "Overview", "/"),
        ("search", "Search Demo", "/?page=search"),
    ]
    chips = []
    for page, label, href in items:
        cls = "chip active" if page == active else "chip"
        chips.append(f'<a class="{cls}" href="{href}">{label}</a>')
    return '<div class="nav">' + "".join(chips) + "</div>"


def _metric_card(label: str, value: str, note: str = "", accent: str = "blue") -> str:
    return f"""
    <div class="metric metric-{accent}">
      <div class="metric-label">{html.escape(label)}</div>
      <div class="metric-value">{html.escape(value)}</div>
      {f'<div class="metric-note">{html.escape(note)}</div>' if note else ''}
    </div>
    """


def _render_overview(report) -> str:
    return f"""
    <section class="hero">
      <div class="eyebrow">Chapter 5 evaluation dashboard</div>
      <h1>Car Search Agent</h1>
      <p class="lede">Separate evaluation view, so the metrics stay clean and easy to present.</p>
      {_nav("overview")}
      <div class="hero-actions">
        <a class="primary-link" href="/?page=search">Open search demo</a>
      </div>
    </section>

    <section class="panel">
      <div class="section-head">
        <h2>Core metrics</h2>
        <p>Overall performance across the dataset and evaluation setup.</p>
      </div>
      <div class="metrics-grid">
        {_metric_card("NLU Hard F1", f"{report['nlu']['hard']['f1']:.3f}", "Exact slot extraction on the required fields", "blue")}
        {_metric_card("NLU Soft F1", f"{report['nlu']['soft']['f1']:.3f}", "Preference detection such as family, luxury, or first owner", "indigo")}
        {_metric_card("NLU Combined F1", f"{report['nlu']['combined']['f1']:.3f}", "Average of hard and soft F1", "emerald")}
        {_metric_card("NLU P/R Mean", f"{report['nlu']['combined']['pr_mean']:.3f}", "Average of precision and recall", "amber")}
        {_metric_card("Smart P@5", f"{report['retrieval']['smart'].precision_at_k:.3f}", "Semantic + rule-based ranking", "indigo")}
        {_metric_card("Baseline P@5", f"{report['retrieval']['baseline'].precision_at_k:.3f}", "Lexical baseline comparator", "amber")}
        {_metric_card("Judge samples", str(len(report['judge_sample'])), "LLM sample reviewed for quality", "amber")}
      </div>
    </section>

    <section class="panel">
      <div class="section-head">
        <h2>Evaluation summary</h2>
        <p>These tables are meant for the presentation and report.</p>
      </div>
      <div class="stacked-panels">
        <div class="subpanel">
          <h3>NLU</h3>
          <table>
            <tr><th>Hard precision</th><td>{report['nlu']['hard']['precision']:.3f}</td></tr>
            <tr><th>Hard recall</th><td>{report['nlu']['hard']['recall']:.3f}</td></tr>
            <tr><th>Hard F1</th><td>{report['nlu']['hard']['f1']:.3f}</td></tr>
            <tr><th>Soft precision</th><td>{report['nlu']['soft']['precision']:.3f}</td></tr>
            <tr><th>Soft recall</th><td>{report['nlu']['soft']['recall']:.3f}</td></tr>
            <tr><th>Soft F1</th><td>{report['nlu']['soft']['f1']:.3f}</td></tr>
            <tr><th>Combined F1</th><td>{report['nlu']['combined']['f1']:.3f}</td></tr>
            <tr><th>Combined P/R mean</th><td>{report['nlu']['combined']['pr_mean']:.3f}</td></tr>
          </table>
        </div>
        <div class="subpanel">
          <h3>Retrieval</h3>
          <table>
            <tr><th>Smart NDCG@5</th><td>{report['retrieval']['smart'].ndcg_at_k:.3f}</td></tr>
            <tr><th>Baseline NDCG@5</th><td>{report['retrieval']['baseline'].ndcg_at_k:.3f}</td></tr>
            <tr><th>No rerank P@5</th><td>{report['retrieval']['no_rerank'].precision_at_k:.3f}</td></tr>
          </table>
        </div>
      </div>
    </section>

    <section class="panel">
      <div class="section-head">
        <h2>Per-query table</h2>
        <p>Use this for the detailed evaluation slide.</p>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Query</th>
              <th>Smart P@5</th>
              <th>Baseline P@5</th>
              <th>Smart NDCG@5</th>
              <th>Baseline NDCG@5</th>
            </tr>
          </thead>
          <tbody>
            {"".join(
                f"<tr><td>{html.escape(row['query'])}</td><td>{row['smart_p@5']:.3f}</td><td>{row['baseline_p@5']:.3f}</td><td>{row['smart_ndcg@5']:.3f}</td><td>{row['baseline_ndcg@5']:.3f}</td></tr>"
                for row in report["case_rows"]
            )}
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <div class="section-head">
        <h2>Ablation and timing</h2>
        <p>Shows the effect of semantic search and reranking.</p>
      </div>
      <div class="stacked-panels">
        <div class="subpanel">
          <h3>Ablation</h3>
          <table>
            <tr><th>No semantic P@5</th><td>{report['ablation']['no_semantic'].precision_at_k:.3f}</td></tr>
            <tr><th>No rerank P@5</th><td>{report['ablation']['no_rerank'].precision_at_k:.3f}</td></tr>
            <tr><th>No rerank NDCG@5</th><td>{report['ablation']['no_rerank'].ndcg_at_k:.3f}</td></tr>
          </table>
        </div>
        <div class="subpanel">
          <h3>Timing</h3>
          <table>
            <tr><th>Smart mean</th><td>{report['timings']['smart'].mean_ms:.2f} ms</td></tr>
            <tr><th>Baseline mean</th><td>{report['timings']['baseline'].mean_ms:.2f} ms</td></tr>
            <tr><th>Smart peak</th><td>{report['timings']['smart'].peak_kb:.1f} KB</td></tr>
            <tr><th>Baseline peak</th><td>{report['timings']['baseline'].peak_kb:.1f} KB</td></tr>
          </table>
        </div>
      </div>
    </section>

    <section class="panel">
      <div class="section-head">
        <h2>Judge sample</h2>
        <p>Short preview of the scored examples.</p>
      </div>
      <div class="sample-list">
        {"".join(
            f"<div class='sample-item'><strong>{html.escape(item['query'])}</strong><span>{html.escape(item['make'])} {html.escape(item['model'])}</span><em>score {item['score']}</em><p>{html.escape(item['rationale'])}</p></div>"
            for item in report["judge_sample"][:10]
        )}
      </div>
    </section>
    """


def _render_search(query: str, results) -> str:
    result_cards = ""
    if results:
        for i, (result, expl) in enumerate(results, 1):
            ad = result.ad
            result_cards += f"""
            <article class="card">
              <div class="card-top">
                <div>
                  <div class="card-kicker">Result #{i}</div>
                  <h3>{html.escape(ad.make)} {html.escape(ad.model)} {ad.year}</h3>
                </div>
                <div class="score-pill">{result.total_score:.3f}</div>
              </div>
              <div class="card-meta">{int(ad.price):,} ₪ | {int(ad.km):,} km | {html.escape(ad.gear_type or '')} | {html.escape(ad.location or '')}</div>
              <p class="card-desc">{html.escape(expl)}</p>
            </article>
            """
    else:
        result_cards = """
        <div class="empty-state">
          Type a query and press Search to see ranked results.
        </div>
        """

    return f"""
    <section class="hero">
      <div class="eyebrow">Live search demo</div>
      <h1>Show results here</h1>
      <p class="lede">This page is only for searches and results, so it is easy to demo on its own.</p>
      {_nav("search")}
      <div class="hero-actions">
        <a class="secondary-link" href="/">Back to evaluations</a>
      </div>
      <form class="search-form" method="get" action="/">
        <input type="hidden" name="page" value="search" />
        <input type="text" name="q" value="{html.escape(query)}" placeholder="Try: Mazda 3 automatic under 70k" />
        <button type="submit">Search</button>
      </form>
      <div class="examples">
        {"".join(
            f'<a class="example-chip" href="/?page=search&q={urllib.parse.quote_plus(ex)}">{html.escape(ex)}</a>'
            for ex in EXAMPLE_QUERIES
        )}
      </div>
    </section>

    <section class="panel">
      <div class="section-head">
        <h2>Results</h2>
        <p>{html.escape(query) if query else 'No query yet.'}</p>
      </div>
      <div class="results-grid">{result_cards}</div>
    </section>
    """


def _render_page(page: str, query: str = "", results=None):
    report = _get_report()
    body_inner = _render_overview(report) if page == "overview" else _render_search(query, results or [])
    return f"""<!doctype html>
<html lang="en" dir="rtl">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Car Search Agent</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --panel: rgba(255,255,255,.82);
      --card: #ffffff;
      --text: #0f172a;
      --muted: #475569;
      --border: #dbe4f0;
      --blue: #2563eb;
      --blue-strong: #1d4ed8;
      --blue-soft: #eff6ff;
      --shadow: 0 12px 40px rgba(15, 23, 42, 0.06);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(37,99,235,.08), transparent 35%),
        radial-gradient(circle at top right, rgba(99,102,241,.08), transparent 30%),
        var(--bg);
      color: var(--text);
    }}
    .wrap {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    .hero {{
      background: linear-gradient(135deg, #eff6ff 0%, #f8fbff 55%, #ffffff 100%);
      border: 1px solid #dbeafe;
      border-radius: 28px;
      padding: 28px;
      box-shadow: var(--shadow);
      margin-bottom: 22px;
    }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: .08em;
      color: #64748b;
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 10px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(32px, 4vw, 46px);
      line-height: 1.05;
    }}
    .lede {{
      margin: 12px 0 0;
      color: var(--muted);
      font-size: 17px;
      max-width: 760px;
      line-height: 1.65;
    }}
    .nav {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 18px;
    }}
    .chip {{
      text-decoration: none;
      color: var(--blue);
      border: 1px solid #bfd7ff;
      background: white;
      padding: 10px 14px;
      border-radius: 999px;
      font-weight: 700;
      transition: .15s ease;
    }}
    .chip:hover {{
      background: var(--blue-soft);
      border-color: var(--blue);
      transform: translateY(-1px);
    }}
    .chip.active {{
      background: var(--blue);
      color: white;
      border-color: var(--blue);
    }}
    .hero-actions {{
      margin-top: 16px;
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .primary-link, .secondary-link {{
      text-decoration: none;
      border-radius: 14px;
      padding: 12px 16px;
      font-weight: 800;
      display: inline-block;
    }}
    .primary-link {{
      background: var(--blue);
      color: white;
      box-shadow: 0 10px 24px rgba(37,99,235,.22);
    }}
    .secondary-link {{
      background: white;
      color: var(--blue);
      border: 1px solid #bfd7ff;
    }}
    .search-form {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 18px;
    }}
    .search-form input[type="text"] {{
      flex: 1;
      min-width: 280px;
      padding: 15px 16px;
      border: 1px solid #cbd5e1;
      border-radius: 16px;
      font-size: 16px;
      background: white;
    }}
    .search-form button {{
      border: 0;
      border-radius: 16px;
      padding: 15px 18px;
      background: var(--blue);
      color: white;
      font-weight: 800;
      cursor: pointer;
    }}
    .examples {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
    }}
    .example-chip {{
      text-decoration: none;
      color: var(--blue-strong);
      background: white;
      border: 1px solid #dbeafe;
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
    }}
    .panel, .subpanel, .metric, .card, .empty-state, .sample-item {{
      background: var(--panel);
      backdrop-filter: blur(10px);
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
    }}
    .panel {{
      border-radius: 24px;
      padding: 22px;
      margin-top: 16px;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: end;
      margin-bottom: 16px;
    }}
    .section-head h2, .section-head h3 {{
      margin: 0;
    }}
    .section-head p {{
      margin: 0;
      color: var(--muted);
    }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .metric {{
      border-radius: 18px;
      padding: 18px 18px 16px;
      min-height: 138px;
      position: relative;
      overflow: hidden;
    }}
    .metric::before {{
      content: "";
      position: absolute;
      inset: 0 0 auto 0;
      height: 5px;
      background: var(--metric-accent, #2563eb);
    }}
    .metric::after {{
      content: "";
      position: absolute;
      right: -24px;
      top: -24px;
      width: 86px;
      height: 86px;
      border-radius: 999px;
      background: var(--metric-glow, rgba(37,99,235,.08));
      pointer-events: none;
    }}
    .metric-blue {{
      --metric-accent: #2563eb;
      --metric-glow: rgba(37,99,235,.10);
    }}
    .metric-indigo {{
      --metric-accent: #4f46e5;
      --metric-glow: rgba(79,70,229,.10);
    }}
    .metric-emerald {{
      --metric-accent: #059669;
      --metric-glow: rgba(5,150,105,.10);
    }}
    .metric-amber {{
      --metric-accent: #d97706;
      --metric-glow: rgba(217,119,6,.10);
    }}
    .metric-label {{
      color: #64748b;
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: .08em;
      line-height: 1.35;
      max-width: 14ch;
    }}
    .metric-value {{
      font-size: clamp(28px, 3vw, 36px);
      font-weight: 900;
      margin-top: 12px;
      line-height: 1;
      letter-spacing: -0.04em;
      word-break: break-word;
    }}
    .metric-note {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
      max-width: 28ch;
    }}
    .split-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .stacked-panels {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
    }}
    .subpanel {{
      border-radius: 18px;
      padding: 16px;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 720px;
      background: transparent;
    }}
    th, td {{
      text-align: right;
      padding: 11px 12px;
      border-bottom: 1px solid #e2e8f0;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{
      background: #f8fafc;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    .results-grid {{
      display: grid;
      gap: 14px;
    }}
    .card {{
      border-radius: 20px;
      padding: 18px;
    }}
    .card-top {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
    }}
    .card-kicker {{
      color: #64748b;
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: .06em;
    }}
    .card h3 {{
      margin: 6px 0 0;
      font-size: 22px;
    }}
    .score-pill {{
      background: var(--blue-soft);
      color: var(--blue-strong);
      border: 1px solid #bfd7ff;
      border-radius: 999px;
      padding: 8px 14px;
      font-weight: 800;
      white-space: nowrap;
    }}
    .card-meta {{
      margin-top: 10px;
      color: #334155;
      font-weight: 700;
    }}
    .card-desc {{
      margin: 10px 0 0;
      color: var(--muted);
      line-height: 1.65;
    }}
    .empty-state {{
      border-radius: 18px;
      padding: 20px;
      color: var(--muted);
    }}
    .sample-list {{
      display: grid;
      gap: 12px;
    }}
    .sample-item {{
      border-radius: 18px;
      padding: 16px;
    }}
    .sample-item strong {{
      display: block;
      margin-bottom: 6px;
    }}
    .sample-item span, .sample-item em {{
      color: var(--muted);
      font-style: normal;
      display: block;
      margin-top: 2px;
    }}
    .sample-item p {{
      margin: 10px 0 0;
      color: #334155;
      line-height: 1.65;
    }}
    @media (max-width: 900px) {{
      .metrics-grid, .split-grid {{
        grid-template-columns: 1fr 1fr;
      }}
      .stacked-panels {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 640px) {{
      .wrap {{
        padding: 16px 12px 40px;
      }}
      .hero, .panel {{
        padding: 18px;
        border-radius: 20px;
      }}
      .metrics-grid, .split-grid, .stacked-panels {{
        grid-template-columns: 1fr;
      }}
      .card-top {{
        flex-direction: column;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    {body_inner}
  </div>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        page = params.get("page", ["overview"])[0].strip() or "overview"
        query = params.get("q", [""])[0].strip()

        results = []
        if page == "search" and query:
            _, results = _smart_search(query)

        body = _render_page(page, query, results).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
