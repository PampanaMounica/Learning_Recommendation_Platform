# src/config.py
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR   = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR    = os.path.join(BASE_DIR, "outputs")

RAW_CSV        = os.path.join(DATA_RAW_DIR,  "courses.csv")
CLEAN_CSV      = os.path.join(DATA_PROC_DIR, "courses_clean.csv")
MATRIX_CSV     = os.path.join(DATA_PROC_DIR, "user_course_matrix.csv")

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_STATE   = 42

# ── Feature Engineering ────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 5000

# ── Recommendation ─────────────────────────────────────────────────────────
TOP_N          = 10          # default number of recommendations

# ── Hybrid Weights ─────────────────────────────────────────────────────────
CONTENT_WEIGHT = 0.5
COLLAB_WEIGHT  = 0.5

# ── Collaborative Filtering ────────────────────────────────────────────────
N_FACTORS      = 100         # SVD latent factors
N_EPOCHS       = 20
LR_ALL         = 0.005
REG_ALL        = 0.02

# ── Evaluation ─────────────────────────────────────────────────────────────
K_VALUES       = [5, 10, 15, 20, 25, 30]