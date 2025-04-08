"""
Configuration module for the Lead Scoring project.
Contains all project-wide constants, paths, and configuration settings.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# File paths - Updated to match actual file locations
RAW_DATA_FILE = PROCESSED_DATA_DIR / "data_aed.csv"  # Raw data file
INTERMEDIATE_DATA_FILE = PROCESSED_DATA_DIR / "data_ptd.csv"  # Processed data file
FINAL_DATA_FILE = RAW_DATA_DIR / "data.csv"  # Final transformed data file

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, TRAINED_MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data processing constants
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Model training constants
HYPERPARAM_GRID = {
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "xgboost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 0.9, 1.0]
    },
    "logistic_regression": {
        "C": [0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }
}

# Feature engineering constants
CATEGORICAL_COLUMNS = [
    "regional",
    "plan_type"
]

NUMERICAL_COLUMNS = [
    "idade_cliente",
    "Qnt_abandono",
    "sum_recharge",
    "recharge_frequency"
]

TARGET_COLUMN = "venda"

# Logging configuration
LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": str(PROJECT_ROOT / "logs" / "app.log"),
            "mode": "a"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
} 