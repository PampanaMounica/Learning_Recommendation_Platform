# src/evaluator.py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from src.config import K_VALUES, OUTPUTS_DIR, RANDOM_STATE
from src.utils import logger, ensure_dirs, set_seed

set_seed(RANDOM_STATE)


def precision_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
    top_k = recommended_ids[:k]
    hits  = sum(1 for i in top_k if i in relevant_ids)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
    top_k = recommended_ids[:k]
    hits  = sum(1 for i in top_k if i in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def f1_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
    p = precision_at_k(recommended_ids, relevant_ids, k)
    r = recall_at_k(recommended_ids, relevant_ids, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def average_precision_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
    score, hits = 0.0, 0
    for i, iid in enumerate(recommended_ids[:k]):
        if iid in relevant_ids:
            hits  += 1
            score += hits / (i + 1)
    return score / min(len(relevant_ids), k) if relevant_ids else 0.0


def ndcg_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
    dcg, idcg = 0.0, 0.0
    for i, iid in enumerate(recommended_ids[:k]):
        if iid in relevant_ids:
            dcg += 1 / np.log2(i + 2)
    for i in range(min(len(relevant_ids), k)):
        idcg += 1 / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


def build_relevant_set(
    user_id    : str,
    ratings_df : pd.DataFrame,
    threshold  : float = 4.0,
) -> set:
    """Courses rated >= threshold by this user = relevant."""
    mask = (
        (ratings_df["user_id"] == user_id) &
        (ratings_df["rating"]  >= threshold)
    )
    return set(ratings_df[mask]["course_id"].tolist())


def evaluate_model(
    model_fn,
    ratings_df : pd.DataFrame,
    df         : pd.DataFrame,
    k_values   : list = K_VALUES,
    n_users    : int  = 50,
) -> pd.DataFrame:
    """
    Evaluate a recommendation function over n_users test users.
    model_fn must accept (user_id) and return a DataFrame with course_id column.
    Returns a DataFrame of mean metrics per K.
    """
    logger.info(f"Evaluating over {n_users} users at K={k_values} ...")
    np.random.seed(RANDOM_STATE)

    user_ids = ratings_df["user_id"].unique()
    sample   = np.random.choice(user_ids, size=min(n_users, len(user_ids)), replace=False)

    rows = []
    for k in k_values:
        p_list, r_list, f1_list, map_list, ndcg_list = [], [], [], [], []
        for uid in sample:
            relevant = build_relevant_set(uid, ratings_df)
            if not relevant:
                continue
            try:
                recs         = model_fn(uid)
                rec_ids      = recs["course_id"].tolist()
                p_list  .append(precision_at_k      (rec_ids, relevant, k))
                r_list  .append(recall_at_k         (rec_ids, relevant, k))
                f1_list .append(f1_at_k             (rec_ids, relevant, k))
                map_list.append(average_precision_at_k(rec_ids, relevant, k))
                ndcg_list.append(ndcg_at_k          (rec_ids, relevant, k))
            except Exception:
                continue

        rows.append({
            "K"          : k,
            "Precision"  : round(np.mean(p_list),    4),
            "Recall"     : round(np.mean(r_list),    4),
            "F1"         : round(np.mean(f1_list),   4),
            "MAP"        : round(np.mean(map_list),  4),
            "NDCG"       : round(np.mean(ndcg_list), 4),
        })

    results_df = pd.DataFrame(rows)
    logger.info("Evaluation complete.")
    return results_df


def plot_metrics(results_df: pd.DataFrame, save: bool = True) -> None:
    ensure_dirs(OUTPUTS_DIR)
    metrics = ["Precision", "Recall", "F1", "MAP", "NDCG"]
    colors  = ["#00d4ff", "#ff6b9d", "#a855f7", "#22c55e", "#f59e0b"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0a0a1a")
    for ax in axes:
        ax.set_facecolor("#0f0f2a")

    # ── Bar chart at K=10 ──────────────────────────────────────────────────
    row_k10 = results_df[results_df["K"] == 10].iloc[0]
    vals    = [row_k10[m] for m in metrics]
    bars    = axes[0].bar(metrics, vals, color=colors, width=0.5, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=9)
    axes[0].set_title("Metric Comparison @ K=10", color="white", fontsize=12, pad=10)
    axes[0].set_ylim(0, 1.1)
    axes[0].tick_params(colors="white")
    axes[0].spines[:].set_color("#333355")

    # ── Precision@K curve ─────────────────────────────────────────────────
    axes[1].plot(results_df["K"], results_df["Precision"],
                 marker="o", color="#00d4ff", linewidth=2, markersize=6)
    for x, y in zip(results_df["K"], results_df["Precision"]):
        axes[1].annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", color="white", fontsize=8)
    axes[1].set_title("Precision@K Curve", color="white", fontsize=12, pad=10)
    axes[1].set_xlabel("K", color="white")
    axes[1].set_ylabel("Precision", color="white")
    axes[1].set_ylim(0, 1.1)
    axes[1].tick_params(colors="white")
    axes[1].spines[:].set_color("#333355")
    axes[1].set_facecolor("#0f0f2a")

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUTS_DIR, "evaluation_metrics.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(f"Plot saved → {path}")
    plt.show()


def print_evaluation_report(results_df: pd.DataFrame) -> None:
    print("\n" + "="*65)
    print("           EVALUATION REPORT")
    print("="*65)
    print(f"  {'K':<6} {'Precision':<12} {'Recall':<12} {'F1':<10} {'MAP':<10} {'NDCG':<10}")
    print("-"*65)
    for _, row in results_df.iterrows():
        print(f"  {int(row.K):<6} {row.Precision:<12.4f} {row.Recall:<12.4f} "
              f"{row.F1:<10.4f} {row.MAP:<10.4f} {row.NDCG:<10.4f}")
    print("="*65)
    best = results_df[results_df["K"] == 10].iloc[0]
    print(f"\n  @ K=10 Summary:")
    print(f"    Precision : {best.Precision:.4f}")
    print(f"    Recall    : {best.Recall:.4f}")
    print(f"    F1-Score  : {best.F1:.4f}")
    print(f"    MAP       : {best.MAP:.4f}")
    print(f"    NDCG      : {best.NDCG:.4f}")
    print("="*65 + "\n")
