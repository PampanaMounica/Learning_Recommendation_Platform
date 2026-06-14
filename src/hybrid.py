# src/hybrid.py
import numpy as np
import pandas as pd
from src.config import TOP_N, CONTENT_WEIGHT, COLLAB_WEIGHT, RANDOM_STATE
from src.utils import logger, set_seed
from src.content_based import get_content_recommendations_by_text
from src.collaborative import get_collaborative_recommendations

set_seed(RANDOM_STATE)


def _normalize_scores(series: pd.Series) -> pd.Series:
    """Min-max normalize a score column to [0, 1]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series * 0 + 1.0
    return (series - mn) / (mx - mn)


def get_hybrid_recommendations(
    user_text     : str,
    user_id       : str,
    vectorizer,
    tfidf_matrix,
    cf_model,
    ratings_df    : pd.DataFrame,
    df            : pd.DataFrame,
    top_n         : int  = TOP_N,
    filter_level  : str  = None,
    filter_platform: str = None,
    content_weight: float = CONTENT_WEIGHT,
    collab_weight : float = COLLAB_WEIGHT,
) -> pd.DataFrame:
    """
    Hybrid recommender:
    - Content score  from TF-IDF cosine similarity on user_text
    - Collab score   from SVD predicted ratings for user_id
    - Final score    = content_weight * content_score + collab_weight * collab_score
    """
    logger.info(f"Running hybrid recommendation for user='{user_id}'")

    # ── Content-based (fetch 3x top_n for better merging pool) ────────────
    cb_recs = get_content_recommendations_by_text(
        user_text      = user_text,
        vectorizer     = vectorizer,
        tfidf_matrix   = tfidf_matrix,
        df             = df,
        top_n          = top_n * 3,
        filter_level   = filter_level,
        filter_platform= filter_platform,
    )

    # ── Collaborative (fetch 3x top_n) ────────────────────────────────────
    cf_recs = get_collaborative_recommendations(
        user_id    = user_id,
        model      = cf_model,
        ratings_df = ratings_df,
        df         = df,
        top_n      = top_n * 3,
    )

    # ── Merge on course_id ─────────────────────────────────────────────────
    merged = pd.merge(
        cb_recs[["course_id", "title", "platform", "level",
                 "rating", "duration", "skills_clean", "content_score"]],
        cf_recs[["course_id", "collab_score"]],
        on  = "course_id",
        how = "outer",
    )

    # Fill missing scores with 0 before normalizing
    merged["content_score"] = merged["content_score"].fillna(0)
    merged["collab_score"]  = merged["collab_score"].fillna(0)

    # ── Normalize then combine ─────────────────────────────────────────────
    merged["content_score_norm"] = _normalize_scores(merged["content_score"])
    merged["collab_score_norm"]  = _normalize_scores(merged["collab_score"])
    merged["hybrid_score"]       = (
        content_weight * merged["content_score_norm"] +
        collab_weight  * merged["collab_score_norm"]
    ).round(4)

    # Re-apply filters after merge (collab might have ignored them)
    if filter_level and filter_level != "All":
        merged = merged[merged["level"] == filter_level]
    if filter_platform and filter_platform != "All":
        merged = merged[merged["platform"] == filter_platform]

    # Fill missing title/platform etc from df using course_id
    missing_mask = merged["title"].isna()
    if missing_mask.any():
        fill = df.set_index("course_id")[["title","platform","level","rating","duration","skills_clean"]]
        merged.loc[missing_mask, ["title","platform","level","rating","duration","skills_clean"]] = (
            merged.loc[missing_mask, "course_id"].map(fill.to_dict("index"))
            .apply(pd.Series)
            .values
        )

    result = (
        merged
        .dropna(subset=["title"])
        .sort_values("hybrid_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return result[[
        "course_id", "title", "platform", "level",
        "rating", "duration", "skills_clean",
        "content_score_norm", "collab_score_norm", "hybrid_score"
    ]]
