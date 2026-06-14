# src/collaborative.py
import os
import pickle
import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from src.config import (
    MODELS_DIR, RANDOM_STATE, TOP_N,
    N_FACTORS, N_EPOCHS, LR_ALL, REG_ALL
)
from src.utils import logger, set_seed

set_seed(RANDOM_STATE)

CF_MODEL_PATH  = os.path.join(MODELS_DIR, "collaborative_model.pkl")
RATINGS_PATH   = os.path.join(MODELS_DIR, "collab_ratings.pkl")


def build_synthetic_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Since we have no real user-course interaction data,
    we synthesize it from rating + popularity signals.
    Creates a user-course ratings DataFrame with 500 simulated users.
    """
    logger.info("Building synthetic user-course ratings ...")
    np.random.seed(RANDOM_STATE)

    n_users  = 500
    n_courses = len(df)

    # Each user rates between 10 and 40 random courses
    rows = []
    for user_id in range(n_users):
        n_rated  = np.random.randint(10, 40)
        # Weight sampling by course rating so popular courses get rated more
        weights  = df["rating"].values / df["rating"].values.sum()
        indices  = np.random.choice(n_courses, size=n_rated, replace=False, p=weights)
        for idx in indices:
            base_rating   = df.iloc[idx]["rating"]
            noise         = np.random.normal(0, 0.3)
            user_rating   = float(np.clip(round(base_rating + noise, 1), 1.0, 5.0))
            rows.append({
                "user_id"   : f"user_{user_id}",
                "course_id" : int(df.iloc[idx]["course_id"]),
                "rating"    : user_rating,
            })

    ratings_df = pd.DataFrame(rows)
    logger.info(f"Synthetic ratings shape: {ratings_df.shape}")
    logger.info(f"Users: {ratings_df['user_id'].nunique()} | Courses rated: {ratings_df['course_id'].nunique()}")
    return ratings_df


def train_collaborative_model(ratings_df: pd.DataFrame):
    """
    Train SVD collaborative filtering model using Surprise library.
    Returns trained model + testset for evaluation.
    """
    logger.info("Training SVD collaborative filtering model ...")

    reader  = Reader(rating_scale=(1.0, 5.0))
    data    = Dataset.load_from_df(
                ratings_df[["user_id", "course_id", "rating"]],
                reader
              )
    trainset, testset = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)

    model = SVD(
        n_factors    = N_FACTORS,
        n_epochs     = N_EPOCHS,
        lr_all       = LR_ALL,
        reg_all      = REG_ALL,
        random_state = RANDOM_STATE,
        verbose      = False,
    )
    model.fit(trainset)

    # Quick evaluation
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae  = accuracy.mae(predictions,  verbose=False)
    logger.info(f"SVD RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    # Save model and ratings
    with open(CF_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(RATINGS_PATH, "wb") as f:
        pickle.dump(ratings_df, f)

    logger.info(f"Collaborative model saved → {CF_MODEL_PATH}")
    return model, testset, rmse, mae


def get_collaborative_recommendations(
    user_id: str,
    model,
    ratings_df: pd.DataFrame,
    df: pd.DataFrame,
    top_n: int = TOP_N,
) -> pd.DataFrame:
    """
    For a known user, predict ratings for all unrated courses
    and return top_n recommendations.
    """
    rated_courses = set(
        ratings_df[ratings_df["user_id"] == user_id]["course_id"].tolist()
    )
    all_course_ids = df["course_id"].tolist()
    unrated        = [cid for cid in all_course_ids if cid not in rated_courses]

    predictions = [model.predict(user_id, cid) for cid in unrated]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_preds   = predictions[:top_n]

    top_ids     = [int(p.iid) for p in top_preds]
    top_scores  = [round(p.est, 3) for p in top_preds]

    results = df[df["course_id"].isin(top_ids)].copy()
    score_map = dict(zip(top_ids, top_scores))
    results["collab_score"] = results["course_id"].map(score_map)
    results = results.sort_values("collab_score", ascending=False)

    return results[
        ["course_id", "title", "platform", "level",
         "rating", "duration", "collab_score"]
    ].reset_index(drop=True)


def load_collaborative_model():
    with open(CF_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(RATINGS_PATH, "rb") as f:
        ratings_df = pickle.load(f)
    logger.info("Collaborative model loaded from disk.")
    return model, ratings_df
