# src/recommender.py
import os
import pickle
import pandas as pd
from scipy.sparse import load_npz
from src.config import MODELS_DIR, DATA_PROC_DIR, TOP_N, RANDOM_STATE
from src.utils import logger, set_seed
from src.content_based import get_content_recommendations_by_text
from src.collaborative import get_collaborative_recommendations, load_collaborative_model
from src.hybrid import get_hybrid_recommendations

set_seed(RANDOM_STATE)

TFIDF_PATH      = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH     = os.path.join(MODELS_DIR, "tfidf_matrix.npz")
FEATURE_DF_PATH = os.path.join(DATA_PROC_DIR, "feature_df.csv")


def load_all_artifacts():
    """Load all saved models and data from disk."""
    with open(TFIDF_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    tfidf_matrix     = load_npz(MATRIX_PATH)
    cf_model, ratings_df = load_collaborative_model()
    df               = pd.read_csv(FEATURE_DF_PATH)
    logger.info("All artifacts loaded successfully.")
    return vectorizer, tfidf_matrix, cf_model, ratings_df, df


def recommend(
    user_text      : str,
    user_id        : str  = "user_0",
    mode           : str  = "hybrid",
    top_n          : int  = TOP_N,
    filter_level   : str  = None,
    filter_platform: str  = None,
) -> pd.DataFrame:
    """
    Unified recommendation API.
    mode: "content" | "collaborative" | "hybrid"
    """
    vectorizer, tfidf_matrix, cf_model, ratings_df, df = load_all_artifacts()

    if mode == "content":
        return get_content_recommendations_by_text(
            user_text      = user_text,
            vectorizer     = vectorizer,
            tfidf_matrix   = tfidf_matrix,
            df             = df,
            top_n          = top_n,
            filter_level   = filter_level,
            filter_platform= filter_platform,
        )
    elif mode == "collaborative":
        return get_collaborative_recommendations(
            user_id    = user_id,
            model      = cf_model,
            ratings_df = ratings_df,
            df         = df,
            top_n      = top_n,
        )
    else:
        return get_hybrid_recommendations(
            user_text      = user_text,
            user_id        = user_id,
            vectorizer     = vectorizer,
            tfidf_matrix   = tfidf_matrix,
            cf_model       = cf_model,
            ratings_df     = ratings_df,
            df             = df,
            top_n          = top_n,
            filter_level   = filter_level,
            filter_platform= filter_platform,
        )
