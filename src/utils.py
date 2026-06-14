# src/utils.py
import os
import random
import numpy as np
import logging

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Global seed set to {seed}")


def ensure_dirs(*paths) -> None:
    """Create directories if they don't already exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)
        logger.info(f"Directory ready: {p}")