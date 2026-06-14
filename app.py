# app.py
import sys
import os
sys.path.insert(0, ".")

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from scipy.sparse import load_npz
from src.config import MODELS_DIR, DATA_PROC_DIR, TOP_N, RANDOM_STATE
from src.utils import set_seed
from src.content_based import get_content_recommendations_by_text
from src.collaborative import get_collaborative_recommendations, load_collaborative_model
from src.hybrid import get_hybrid_recommendations

set_seed(RANDOM_STATE)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Learning Recommender",
    page_icon  = "🎓",
    layout     = "wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    body { background-color: #0a0a1a; }
    .main { background-color: #0a0a1a; }
    .stApp { background-color: #0a0a1a; }
    h1, h2, h3, p, label { color: white !important; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a3e, #0f0f2a);
        border: 1px solid #3333aa;
        border-radius: 12px;
        padding: 16px;
        margin: 6px 0;
    }
    .course-card {
        background: linear-gradient(135deg, #0f0f2a, #1a1a3e);
        border: 1px solid #4444cc;
        border-radius: 14px;
        padding: 18px;
        margin: 10px 0;
    }
    .platform-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin-right: 6px;
    }
    .score-bar {
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(90deg, #00d4ff, #a855f7);
        margin-top: 6px;
    }
    div[data-testid="stSelectbox"] label { color: white !important; }
    div[data-testid="stTextArea"] label  { color: white !important; }
    div[data-testid="stSlider"]   label  { color: white !important; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts (cached) ────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    tfidf_path  = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    matrix_path = os.path.join(MODELS_DIR, "tfidf_matrix.npz")
    df_path     = os.path.join(DATA_PROC_DIR, "feature_df.csv")

    with open(tfidf_path, "rb") as f:
        vectorizer = pickle.load(f)
    tfidf_matrix         = load_npz(matrix_path)
    cf_model, ratings_df = load_collaborative_model()
    df                   = pd.read_csv(df_path)
    df["course_id"]      = df["course_id"].astype(int)
    return vectorizer, tfidf_matrix, cf_model, ratings_df, df


# ── Platform badge colors ──────────────────────────────────────────────────
PLATFORM_COLORS = {
    "Coursera"   : "#0056d2",
    "Udemy"      : "#a435f0",
    "edX"        : "#02262b",
    "Skillshare" : "#00e676",
}

def platform_badge(platform):
    color = PLATFORM_COLORS.get(platform, "#555577")
    return f'<span class="platform-badge" style="background:{color};color:white;">{platform}</span>'


def stars(rating):
    full  = int(round(rating))
    full  = max(0, min(5, full))
    return "★" * full + "☆" * (5 - full)


def score_bar(score, max_score=1.0):
    pct = min(int((score / max_score) * 100), 100)
    return f'<div class="score-bar" style="width:{pct}%;"></div>'


def render_course_card(row, idx, score_col, score_label):
    score    = row.get(score_col, 0)
    rating   = row.get("rating", 0)
    platform = row.get("platform", "Unknown")
    level    = row.get("level",    "Unknown")
    duration = row.get("duration", "Unknown")
    title    = row.get("title",    "Unknown")
    skills   = str(row.get("skills_clean", ""))[:120]

    st.markdown(f"""
    <div class="course-card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="color:#00d4ff; font-size:13px; font-weight:bold;">#{idx+1}</span>
            {platform_badge(platform)}
            <span style="color:#a855f7; font-size:12px;">{level}</span>
        </div>
        <h4 style="color:white; margin:8px 0 4px 0; font-size:15px;">{title}</h4>
        <div style="color:#fbbf24; font-size:14px;">{stars(rating)} <span style="color:#aaa; font-size:12px;">({rating:.1f})</span></div>
        <div style="color:#888; font-size:12px; margin:4px 0;">⏱ {duration}</div>
        <div style="color:#ccc; font-size:11px; margin:4px 0;">🏷 {skills}...</div>
        <div style="margin-top:8px;">
            <span style="color:#00d4ff; font-size:12px;">{score_label}: {score:.4f}</span>
            {score_bar(score)}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Main App ───────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div style="text-align:center; padding:20px 0 10px 0;">
        <h1 style="font-size:2.8em; background:linear-gradient(90deg,#00d4ff,#a855f7,#ff6b9d);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            🎓 Personalized Learning Recommender
        </h1>
        <p style="color:#888; font-size:1em;">
            ML-powered course recommendations from Coursera · Udemy · edX · Skillshare
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load
    with st.spinner("Loading models..."):
        vectorizer, tfidf_matrix, cf_model, ratings_df, df = load_artifacts()

    

    st.markdown("---")

    # ── Sidebar controls ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        mode = st.selectbox(
            "Recommendation Mode",
            ["Hybrid (Best)", "Content-Based", "Collaborative"],
        )

        user_id = st.selectbox(
            "Select User ID",
            options=sorted(ratings_df["user_id"].unique())[:50],
            index=0,
        )

        top_n = st.slider("Number of Recommendations", 5, 20, 10)

        filter_level = st.selectbox(
            "Filter by Level",
            ["All", "Beginner", "Intermediate", "Advanced"],
        )

        filter_platform = st.selectbox(
            "Filter by Platform",
            ["All", "Coursera", "Udemy", "edX", "Skillshare"],
        )

        st.markdown("---")
        st.markdown("### 📊 Dataset Stats")
        st.markdown(f"- **Total Courses:** {len(df):,}")
        for p, c in df["platform"].value_counts().items():
            st.markdown(f"- **{p}:** {c:,}")

    # ── Main input ─────────────────────────────────────────────────────────
    st.markdown("### 🔍 What do you want to learn?")
    user_text = st.text_area(
        label       = "Enter your skills, interests or career goals:",
        placeholder = "e.g. Python machine learning data science neural networks",
        height      = 100,
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        run_btn = st.button("🚀 Get Recommendations", use_container_width=True)
    with col2:
        clear_btn = st.button("🔄 Clear", use_container_width=True)

    if clear_btn:
        st.rerun()

    # ── Results ────────────────────────────────────────────────────────────
    if run_btn:
        if not user_text.strip() and mode != "Collaborative":
            st.warning("Please enter something you want to learn!")
            return

        fl = None if filter_level    == "All" else filter_level
        fp = None if filter_platform == "All" else filter_platform

        with st.spinner("Finding best courses for you..."):

            if mode == "Content-Based":
                recs = get_content_recommendations_by_text(
                    user_text      = user_text,
                    vectorizer     = vectorizer,
                    tfidf_matrix   = tfidf_matrix,
                    df             = df,
                    top_n          = top_n,
                    filter_level   = fl,
                    filter_platform= fp,
                )
                score_col, score_label = "content_score", "Content Score"

            elif mode == "Collaborative":
                recs = get_collaborative_recommendations(
                    user_id    = user_id,
                    model      = cf_model,
                    ratings_df = ratings_df,
                    df         = df,
                    top_n      = top_n,
                )
                score_col, score_label = "collab_score", "Predicted Rating"

            else:
                recs = get_hybrid_recommendations(
                    user_text      = user_text,
                    user_id        = user_id,
                    vectorizer     = vectorizer,
                    tfidf_matrix   = tfidf_matrix,
                    cf_model       = cf_model,
                    ratings_df     = ratings_df,
                    df             = df,
                    top_n          = top_n,
                    filter_level   = fl,
                    filter_platform= fp,
                )
                score_col, score_label = "hybrid_score", "Hybrid Score"

        st.markdown(f"### 🎯 Top {len(recs)} Recommendations — *{mode}*")

        # ── Summary metrics row ────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Courses Found",  len(recs))
        m2.metric("Avg Rating",     f"{recs['rating'].mean():.2f}" if "rating" in recs.columns else "N/A")
        m3.metric("Top Score",      f"{recs[score_col].max():.4f}" if score_col in recs.columns else "N/A")
        m4.metric("Platforms",      recs["platform"].nunique() if "platform" in recs.columns else "N/A")

        st.markdown("---")

        # ── Course cards ───────────────────────────────────────────────────
        for idx, row in recs.iterrows():
            render_course_card(row, idx, score_col, score_label)

        # ── Download button ────────────────────────────────────────────────
        st.markdown("---")
        csv = recs.to_csv(index=False)
        st.download_button(
            label     = "📥 Download Recommendations as CSV",
            data      = csv,
            file_name = "recommendations.csv",
            mime      = "text/csv",
        )


if __name__ == "__main__":
    main()
