import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# FastF1 settings
CACHE_DIR = BASE_DIR / "f1_cache"

# Model settings
DEFAULT_MODEL_PARAMS = {
    "GradientBoostingRegressor": {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "random_state": 42
    },
    "RandomForestRegressor": {
        "n_estimators": 100,
        "random_state": 42
    }
}

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, CACHE_DIR]:
    directory.mkdir(exist_ok=True)