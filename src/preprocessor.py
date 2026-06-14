# src/preprocessor.py
import re
import pandas as pd
import numpy as np
from src.config import RANDOM_STATE, CLEAN_CSV, DATA_PROC_DIR
from src.utils import logger, ensure_dirs, set_seed

set_seed(RANDOM_STATE)

LEVEL_MAP = {
    "beginner"            : "Beginner",
    "beginner level"      : "Beginner",
    "easy"                : "Beginner",
    "intermediate"        : "Intermediate",
    "intermediate level"  : "Intermediate",
    "medium"              : "Intermediate",
    "advanced"            : "Advanced",
    "advanced level"      : "Advanced",
    "hard"                : "Advanced",
    "expert"              : "Advanced",
    "all"                 : "All Levels",
    "all levels"          : "All Levels",
    "appropriate for all" : "All Levels",
}

def _drop_bad_rows(df):
    before = len(df)
    df = df.dropna(subset=["title"])
    df = df[df["title"].str.strip().ne("")]
    df = df.drop_duplicates(subset=["title", "platform"])
    after = len(df)
    logger.info(f"Dropped {before - after:,} bad/duplicate rows → {after:,} remain")
    return df.reset_index(drop=True)

def _clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\{|\}", "", text)
    text = re.sub(r'\"', "", text)
    text = re.sub(r"[^a-zA-Z0-9\s,\.\-\+\#]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _build_skills(df):
    def combine_row(row):
        parts = []
        for col in ["skills", "associatedskills", "subject", "description"]:
            val = row.get(col, "")
            cleaned = _clean_text(str(val) if pd.notna(val) else "")
            if cleaned:
                parts.append(cleaned)
        return " , ".join(parts)
    df["skills_clean"] = df.apply(combine_row, axis=1)
    logger.info("Built skills_clean")
    return df

def _normalize_level(df):
    def map_level(val):
        if pd.isna(val):
            return "All Levels"
        key = str(val).strip().lower()
        return LEVEL_MAP.get(key, "All Levels")
    df["level_clean"] = df["level"].apply(map_level)
    logger.info(f"Level distribution:\n{df['level_clean'].value_counts().to_string()}")
    return df

def _normalize_rating(df):
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    global_median = df["rating"].median()
    df["rating_clean"] = df.groupby("platform")["rating"].transform(
        lambda x: x.fillna(x.median() if x.notna().any() else global_median)
    )
    df["rating_clean"] = df["rating_clean"].fillna(global_median).round(2)
    logger.info(f"Rating median: {global_median:.2f}")
    return df

def _normalize_duration(df):
    df["duration_clean"] = df["duration"].fillna("Unknown").astype(str).str.strip()
    return df

def _clean_title(df):
    df["title_clean"] = df["title"].apply(_clean_text)
    return df

def _build_text_features(df):
    df["text_features"] = (
        df["title_clean"].fillna("") + " " +
        df["skills_clean"].fillna("") + " " +
        df["level_clean"].fillna("") + " " +
        df["platform"].fillna("")
    )
    df["text_features"] = df["text_features"].str.replace(r"\s+", " ", regex=True).str.strip()
    logger.info("Built text_features")
    return df

FINAL_COLS = [
    "title_clean", "platform", "level_clean",
    "rating_clean", "duration_clean",
    "skills_clean", "text_features",
    "provider", "description"
]

def _select_final_columns(df):
    available = [c for c in FINAL_COLS if c in df.columns]
    df = df[available].copy()
    df = df.rename(columns={
        "title_clean"    : "title",
        "level_clean"    : "level",
        "rating_clean"   : "rating",
        "duration_clean" : "duration",
    })
    df = df.reset_index(drop=True)
    df["course_id"] = df.index
    return df

def preprocess(df):
    logger.info("=== Starting Preprocessing Pipeline ===")
    df = _drop_bad_rows(df)
    df = _clean_title(df)
    df = _build_skills(df)
    df = _normalize_level(df)
    df = _normalize_rating(df)
    df = _normalize_duration(df)
    df = _build_text_features(df)
    df = _select_final_columns(df)
    ensure_dirs(DATA_PROC_DIR)
    df.to_csv(CLEAN_CSV, index=False)
    logger.info(f"Saved → {CLEAN_CSV} | Shape: {df.shape}")
    return df

def print_summary(df):
    print("\n" + "="*55)
    print("       PREPROCESSING SUMMARY")
    print("="*55)
    print(f"  Total clean courses : {len(df):,}")
    print(f"  Columns             : {list(df.columns)}")
    print(f"\n  Platform breakdown:")
    for p, c in df["platform"].value_counts().items():
        print(f"    {p:<15} → {c:>6,}")
    print(f"\n  Level breakdown:")
    for l, c in df["level"].value_counts().items():
        print(f"    {l:<20} → {c:>6,}")
    print(f"\n  Rating stats:")
    print(f"    Mean   : {df['rating'].mean():.2f}")
    print(f"    Median : {df['rating'].median():.2f}")
    print(f"    Min    : {df['rating'].min():.2f}")
    print(f"    Max    : {df['rating'].max():.2f}")
    print(f"\n  Text features sample (row 0):")
    print(f"    {df['text_features'].iloc[0][:120]}...")
    print("="*55 + "\n")
