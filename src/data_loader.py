# src/data_loader.py
import os
import pandas as pd
from src.config import DATA_RAW_DIR
from src.utils import logger

# ── Per-platform column maps → all renamed to standard names ──────────────
PLATFORM_COLUMN_MAPS = {
    "Coursera": {
        "course"      : "title",
        "skills"      : "skills",
        "rating"      : "rating",
        "reviewcount" : "reviews",
        "level"       : "level",
        "duration"    : "duration",
        "partner"     : "provider",
        "certificatetype": "certificate",
    },
    "Udemy": {
        "course_title"          : "title",
        "subject"               : "category",
        "content_duration"      : "duration",
        "avg_rating"            : "rating",
        "num_reviews"           : "reviews",
        "level"                 : "level",
        "is_paid"               : "is_paid",
        "price"                 : "price",
        "num_subscribers"       : "subscribers",
    },
    "edX": {
        "course_title"  : "title",
        "course_description": "description",
        "skills"        : "skills",
        "level"         : "level",
        "course_rating" : "rating",
        "duration"      : "duration",
        "institution"   : "provider",
        "course_type"   : "certificate",
    },
    "Skillshare": {
        "title"         : "title",
        "instructor"    : "provider",
        "rating"        : "rating",
        "reviews"       : "reviews",
        "duration"      : "duration",
        "skills"        : "skills",
        "level"         : "level",
        "category"      : "category",
    },
}

# ── Filenames to look for ─────────────────────────────────────────────────
PLATFORM_FILES = {
    "Coursera"   : "Coursera.csv",
    "Udemy"      : "Udemy.csv",
    "edX"        : "edx.csv",
    "Skillshare" : "skillshare.csv",
}


def _load_single_platform(platform: str, filepath: str) -> pd.DataFrame:
    """Load one platform CSV, rename columns, add platform tag."""
    logger.info(f"Loading {platform} from {filepath}")
    df = pd.read_csv(filepath, low_memory=False)

    # lowercase all column names for safe matching
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # build rename map using lowercased keys
    col_map = PLATFORM_COLUMN_MAPS.get(platform, {})
    col_map_lower = {k.lower(): v for k, v in col_map.items()}

    df = df.rename(columns=col_map_lower)
    df["platform"] = platform

    logger.info(f"  {platform}: {len(df):,} rows | columns: {list(df.columns)}")
    return df


def load_raw_data() -> pd.DataFrame:
    """
    Load all 4 platform CSVs from data/raw/,
    merge into one unified DataFrame.
    """
    frames = []

    for platform, filename in PLATFORM_FILES.items():
        filepath = os.path.join(DATA_RAW_DIR, filename)
        if not os.path.exists(filepath):
            logger.warning(f"File not found, skipping: {filepath}")
            continue
        df = _load_single_platform(platform, filepath)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No CSV files found in data/raw/.\n"
            "Please place Coursera.csv, Udemy.csv, edx.csv, skillshare.csv there."
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    logger.info(f"Combined dataset shape: {combined.shape}")
    return combined


def validate_data(df: pd.DataFrame) -> None:
    """Print a health-check report."""
    print("\n" + "="*55)
    print("         DATA VALIDATION REPORT")
    print("="*55)
    print(f"  Total courses     : {len(df):,}")
    print(f"  Total columns     : {df.shape[1]}")

    print("\n  Courses per platform:")
    for platform, count in df["platform"].value_counts().items():
        print(f"    {platform:<15} → {count:>6,} courses")

    key_cols = ["title", "skills", "rating", "level", "duration", "platform"]
    print("\n  Missing values in key columns:")
    for col in key_cols:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            pct = (null_count / len(df)) * 100
            print(f"    {col:<15} → {null_count:>5,} missing  ({pct:.1f}%)")

    print("\n  Sample rows (first 3):")
    show_cols = [c for c in key_cols if c in df.columns]
    print(df[show_cols].head(3).to_string(index=False))
    print("="*55 + "\n")