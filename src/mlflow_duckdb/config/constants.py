"""Constants used throughout the MLflow DuckDB tracking store implementation.

This module centralizes all static configuration values, such as environment
variable keys, default IDs, error messages, tag keys, and numeric limits.
"""

# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────
ENV_READ_ONLY = "MLFLOW_DUCKDB_READ_ONLY"
TRUTHY_STRINGS = {"1", "true", "yes"}

# ─────────────────────────────────────────────
# Default Experiment
# ─────────────────────────────────────────────
DEFAULT_EXPERIMENT_ID = "0"
DEFAULT_EXPERIMENT_NAME = "Default"
EXPERIMENT_ARTIFACT_PATH_TEMPLATE = "{db_path}/artifacts/{experiment_id}"

# ─────────────────────────────────────────────
# Default Paths
# ─────────────────────────────────────────────
DEFAULT_DB_FILENAME = "~/mlruns.duckdb"

# ─────────────────────────────────────────────
# MLflow Tag Keys
# ─────────────────────────────────────────────
MLFLOW_RUN_NAME_TAG = "mlflow.runName"
MLFLOW_DATASET_CONTEXT = "mlflow.context"  # Used to indicate dataset context

# ─────────────────────────────────────────────
# Limits
# ─────────────────────────────────────────────
DEFAULT_MAX_EXPERIMENT_RESULTS = 1000
MAX_DATASET_SUMMARIES_RESULTS = 1000

# ─────────────────────────────────────────────
# Time
# ─────────────────────────────────────────────
MS_IN_SECOND = 1000  # Number of milliseconds in one second

# ─────────────────────────────────────────────
# Error Messages
# ─────────────────────────────────────────────
ERR_MSG_INVALID_URI = (
    "Invalid DuckDB store URI: {uri}. Must start with "
    "'duckdb://', 'duckdb:///', or 'duckdb:'."
)

ERR_MSG_READ_ONLY_FILE_MISSING = (
    "Cannot open database '{db_path}' in read-only mode: database does not exist. "
    "Ensure the database file exists before launching MLflow UI in read-only mode."
)

ERR_MSG_RUN_NOT_FOUND = "Run '{run_id}' not found"
