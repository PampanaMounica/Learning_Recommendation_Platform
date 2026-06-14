# src/content_based.py
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz
from src.config import MODELS_DIR, TOP_N, RANDOM_STATE
from src.utils import logger, set_seed

set_seed(RANDOM_STATE)

TFIDF_PATH  = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH = os.path.join(MODELS_DIR, "tfidf_matrix.npz")
CB_MODEL_PATH = os.path.join(MODELS_DIR, "content_based_model.pkl")


def load_tfidf_artifacts():
    with open(TFIDF_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    tfidf_matrix = load_npz(MATRIX_PATH)
    logger.info(f"TF-IDF loaded → shape: {tfidf_matrix.shape}")
    return vectorizer, tfidf_matrix


def build_content_based_model(tfidf_matrix):
    """
    Precompute cosine similarity in chunks to avoid memory crash
    on 40k+ courses. Saves the full similarity model.
    """
    logger.info("Building content-based similarity model ...")
    model = {
        "tfidf_matrix" : tfidf_matrix,
        "type"         : "cosine"
    }
    with open(CB_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Content-based model saved → {CB_MODEL_PATH}")
    return model


def get_content_recommendations(
    query_idx: int,
    tfidf_matrix,
    df: pd.DataFrame,
    top_n: int = TOP_N,
    filter_level: str = None,
    filter_platform: str = None,
) -> pd.DataFrame:
    """
    Given a course index, return top_n similar courses
    using cosine similarity on TF-IDF matrix.
    """
    query_vec = tfidf_matrix[query_idx]
    scores    = cosine_similarity(query_vec, tfidf_matrix).flatten()
    scores[query_idx] = -1          # exclude the query course itself

    top_indices = np.argsort(scores)[::-1]
    results     = df.iloc[top_indices].copy()
    results["content_score"] = scores[top_indices]

    # ── Optional filters ───────────────────────────────────────────────────
    if filter_level and filter_level != "All":
        results = results[results["level"] == filter_level]
    if filter_platform and filter_platform != "All":
        results = results[results["platform"] == filter_platform]

    return results.head(top_n)[
        ["course_id", "title", "platform", "level",
         "rating", "duration", "skills_clean", "content_score"]
    ].reset_index(drop=True)


def get_content_recommendations_by_text(
    user_text: str,
    vectorizer,
    tfidf_matrix,
    df: pd.DataFrame,
    top_n: int = TOP_N,
    filter_level: str = None,
    filter_platform: str = None,
) -> pd.DataFrame:
    """
    Given free-text input (skills/interests), transform with TF-IDF
    and return top_n matching courses by cosine similarity.
    """
    query_vec = vectorizer.transform([user_text])
    scores    = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(scores)[::-1]

    results = df.iloc[top_indices].copy()
    results["content_score"] = scores[top_indices]

    if filter_level and filter_level != "All":
        results = results[results["level"] == filter_level]
    if filter_platform and filter_platform != "All":
        results = results[results["platform"] == filter_platform]

    return results.head(top_n)[
        ["course_id", "title", "platform", "level",
         "rating", "duration", "skills_clean", "content_score"]
    ].reset_index(drop=True)
