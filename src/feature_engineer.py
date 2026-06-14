# src/feature_engineer.py
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, save_npz, load_npz
from src.config import (
    RANDOM_STATE, TFIDF_MAX_FEATURES,
    MODELS_DIR, DATA_PROC_DIR, CLEAN_CSV
)
from src.utils import logger, ensure_dirs, set_seed

set_seed(RANDOM_STATE)

# ── File paths ─────────────────────────────────────────────────────────────
TFIDF_PATH       = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MATRIX_PATH      = os.path.join(MODELS_DIR, "tfidf_matrix.npz")
SCALER_PATH      = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_DF_PATH  = os.path.join(DATA_PROC_DIR, "feature_df.csv")


# ── 1. TF-IDF on text_features ─────────────────────────────────────────────
def build_tfidf(df: pd.DataFrame):
    """
    Fit TF-IDF on the text_features column.
    Returns: (tfidf_matrix, vectorizer)
    """
    logger.info(f"Fitting TF-IDF (max_features={TFIDF_MAX_FEATURES}) ...")
    vectorizer = TfidfVectorizer(
        max_features = TFIDF_MAX_FEATURES,
        ngram_range  = (1, 2),
        stop_words   = "english",
        sublinear_tf = True,
    )
    tfidf_matrix = vectorizer.fit_transform(df["text_features"].fillna(""))
    logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    return tfidf_matrix, vectorizer


# ── 2. Numeric features (rating + level encoded) ───────────────────────────
def build_numeric_features(df: pd.DataFrame):
    """
    Scale rating and encode level as numeric.
    Returns: scaled numpy array (n_courses x 2)
    """
    level_order = {"Beginner": 0, "Intermediate": 1, "Advanced": 2, "All Levels": 1}
    df["level_encoded"] = df["level"].map(level_order).fillna(1)

    scaler = MinMaxScaler()
    numeric = scaler.fit_transform(df[["rating", "level_encoded"]].fillna(0))
    logger.info(f"Numeric features shape: {numeric.shape}")
    return numeric, scaler


# ── 3. Platform one-hot ────────────────────────────────────────────────────
def build_platform_features(df: pd.DataFrame):
    """One-hot encode platform column."""
    platform_dummies = pd.get_dummies(df["platform"], prefix="platform")
    logger.info(f"Platform features shape: {platform_dummies.shape}")
    return platform_dummies.values


# ── 4. Combine all features ────────────────────────────────────────────────
def build_combined_matrix(tfidf_matrix, numeric_features, platform_features):
    """
    Stack TF-IDF (sparse) + numeric + platform into one sparse matrix.
    """
    from scipy.sparse import csr_matrix
    numeric_sparse  = csr_matrix(numeric_features)
    platform_sparse = csr_matrix(platform_features)
    combined = hstack([tfidf_matrix, numeric_sparse, platform_sparse])
    logger.info(f"Combined feature matrix shape: {combined.shape}")
    return combined


# ── 5. Save all artifacts ──────────────────────────────────────────────────
def save_artifacts(vectorizer, scaler, tfidf_matrix, combined_matrix):
    ensure_dirs(MODELS_DIR)
    with open(TFIDF_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    save_npz(MATRIX_PATH, tfidf_matrix)
    logger.info(f"TF-IDF vectorizer saved → {TFIDF_PATH}")
    logger.info(f"TF-IDF matrix saved     → {MATRIX_PATH}")
    logger.info(f"Scaler saved            → {SCALER_PATH}")


# ── 6. Load artifacts ──────────────────────────────────────────────────────
def load_artifacts():
    with open(TFIDF_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    tfidf_matrix = load_npz(MATRIX_PATH)
    logger.info("Artifacts loaded from disk.")
    return vectorizer, scaler, tfidf_matrix


# ── Master function ────────────────────────────────────────────────────────
def run_feature_engineering(df: pd.DataFrame):
    """
    Full pipeline: build TF-IDF + numeric + platform features,
    combine them, save to disk, return everything needed downstream.
    """
    logger.info("=== Starting Feature Engineering ===")

    tfidf_matrix, vectorizer  = build_tfidf(df)
    numeric_features, scaler  = build_numeric_features(df)
    platform_features         = build_platform_features(df)
    combined_matrix           = build_combined_matrix(
                                    tfidf_matrix,
                                    numeric_features,
                                    platform_features
                                )
    save_artifacts(vectorizer, scaler, tfidf_matrix, combined_matrix)

    # Save feature-enriched df
    df["level_encoded"] = df["level"].map(
        {"Beginner": 0, "Intermediate": 1, "Advanced": 2, "All Levels": 1}
    ).fillna(1)
    ensure_dirs(DATA_PROC_DIR)
    df.to_csv(FEATURE_DF_PATH, index=False)
    logger.info(f"Feature df saved → {FEATURE_DF_PATH}")
    logger.info("=== Feature Engineering Complete ===")

    return {
        "df"              : df,
        "tfidf_matrix"    : tfidf_matrix,
        "combined_matrix" : combined_matrix,
        "vectorizer"      : vectorizer,
        "scaler"          : scaler,
    }


def print_feature_summary(artifacts: dict):
    df     = artifacts["df"]
    tfidf  = artifacts["tfidf_matrix"]
    combined = artifacts["combined_matrix"]
    print("\n" + "="*55)
    print("      FEATURE ENGINEERING SUMMARY")
    print("="*55)
    print(f"  Courses              : {len(df):,}")
    print(f"  TF-IDF matrix        : {tfidf.shape}")
    print(f"  Combined matrix      : {combined.shape}")
    print(f"  Top TF-IDF terms (10): {artifacts['vectorizer'].get_feature_names_out()[:10].tolist()}")
    print("="*55 + "\n")
