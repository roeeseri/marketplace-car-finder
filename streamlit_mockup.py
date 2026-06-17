from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="Marketplace Car Finder Agent",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f7fafc 0%, #eef4f9 100%);
            font-family: "Segoe UI", Arial, sans-serif;
        }
        .hero {
            background: white;
            padding: 2rem;
            border-radius: 24px;
            border: 1px solid #e5e7eb;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin-bottom: 0.25rem;
            color: #0f172a;
            font-size: 2.2rem;
        }
        .hero p {
            color: #475569;
            margin-top: 0.25rem;
            margin-bottom: 0;
            font-size: 1rem;
        }
        .section-title {
            font-size: 1.2rem;
            font-weight: 700;
            margin: 1rem 0 0.6rem 0;
            color: #0f172a;
        }
        .result-card {
            background: white;
            border: 1px solid #dbe4ee;
            border-radius: 20px;
            padding: 1.1rem 1.1rem 0.9rem 1.1rem;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
            margin-bottom: 0.8rem;
        }
        .car-title {
            font-size: 1.15rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .car-meta {
            color: #475569;
            font-size: 0.95rem;
            margin-bottom: 0.7rem;
        }
        .price-badge {
            display: inline-block;
            background: #e8f5e9;
            color: #166534;
            border: 1px solid #bbf7d0;
            border-radius: 999px;
            padding: 0.22rem 0.65rem;
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }
        .explain-box {
            background: #f8fbff;
            border: 1px solid #dbeafe;
            border-radius: 16px;
            padding: 0.8rem 0.9rem;
            color: #1e3a8a;
            font-size: 0.95rem;
            line-height: 1.55;
        }
        .note {
            color: #64748b;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Marketplace Car Finder Agent</h1>
        <p>סוכן חכם לחיפוש רכבים יד שנייה באמצעות NLU, חיפוש סמנטי, דירוג רב-קריטריוני והסבר תוצאות.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

query = st.text_input(
    "חיפוש חופשי",
    value="אני מחפש רכב קטן לסטודנט עד 40 אלף, אוטומט, אמין וחסכוני",
)
st.button("Search", type="primary")

st.markdown('<div class="section-title">תוצאות חיפוש</div>', unsafe_allow_html=True)
st.markdown('<div class="note">הדוגמה הבאה מיועדת לצילום מסך איכותי של ממשק המערכת.</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        """
        <div class="result-card">
            <div class="car-title">Mazda 2</div>
            <div class="car-meta">שנת 2018 · 38,000 ק"מ · אוטומט · תל אביב</div>
            <div class="price-badge">38,000 ILS</div>
            <div class="explain-box">
                <strong>Explainability:</strong> מדוע רכב זה מתאים לך? עומד בתקציב, אוטומט,
                חסכוני בדלק, ומספק איזון טוב בין אמינות, עלות ותחזוקה.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="result-card">
            <div class="car-title">Toyota Yaris</div>
            <div class="car-meta">שנת 2019 · 42,000 ק"מ · אוטומט · חיפה</div>
            <div class="price-badge">39,500 ILS</div>
            <div class="explain-box">
                <strong>Explainability:</strong> מדוע רכב זה מתאים לך? עומד בתקציב, אוטומט,
                אמין מאוד, חסכוני בדלק, ומתאים במיוחד לסטודנט שצריך רכב עירוני.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption("Mockup screenshot for the final engineering project book.")
