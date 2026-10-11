"""Loading, validating and describing the datasets."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from scipy import stats

from fifa_regression.config import (
    MAX_SHARED_VARIABLES,
    N_EXPLANATORY_VARIABLES,
    SHARED_CONCEPTS,
    ModelConfig,
)

logger = logging.getLogger(__name__)


def resolve_data_path(cfg: ModelConfig, data_dir: Path) -> Path:
    """Return the first existing file from ``cfg.filenames`` inside ``data_dir``."""
    for name in cfg.filenames:
        candidate = Path(data_dir) / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"None of {cfg.filenames} found in {data_dir}")


def load_dataset(cfg: ModelConfig, data_dir: Path) -> pd.DataFrame:
    """Read the CSV for one objective and check it meets the project rules."""
    path = resolve_data_path(cfg, data_dir)
    df = pd.read_csv(path)
    logger.info("Loaded %s: %d rows x %d columns", path.name, *df.shape)
    validate_dataset(df, cfg)
    return df


def validate_dataset(df: pd.DataFrame, cfg: ModelConfig) -> None:
    """Fail fast if the dataset breaks a project requirement."""
    missing_cols = [c for c in (cfg.target, *cfg.features) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"[{cfg.tag}] missing columns: {missing_cols}")
    if len(cfg.features) != N_EXPLANATORY_VARIABLES:
        raise ValueError(f"[{cfg.tag}] expected {N_EXPLANATORY_VARIABLES} explanatory "
                         f"variables, got {len(cfg.features)}")
    overlap = set(cfg.features) & set(cfg.id_columns)
    if overlap:
        raise ValueError(f"[{cfg.tag}] ID columns used as predictors: {overlap}")
    n_missing = int(df[[cfg.target, *cfg.features]].isna().sum().sum())
    if n_missing:
        raise ValueError(f"[{cfg.tag}] {n_missing} missing values in model columns")


def check_shared_variables() -> int:
    """Confirm no more than four concepts are shared between 2.1 and 2.2."""
    n_shared = len(SHARED_CONCEPTS)
    if n_shared > MAX_SHARED_VARIABLES:
        raise ValueError(f"{n_shared} shared variables (max {MAX_SHARED_VARIABLES})")
    return n_shared


def quality_report(df: pd.DataFrame) -> dict:
    """Basic data-quality counts."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def describe(df: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """Descriptive statistics with skewness and Shapiro-Wilk p-value per column."""
    cols = [cfg.target, *cfg.features]
    table = df[cols].describe().T
    table["skew"] = df[cols].apply(stats.skew)
    table["shapiro_p"] = df[cols].apply(lambda s: stats.shapiro(s).pvalue)
    return table


def check_consistency(df_21: pd.DataFrame, df_22: pd.DataFrame) -> int:
    """Count matches where 2.1 goal difference disagrees with 2.2 goals (should be 0)."""
    mismatches = 0
    for _, row in df_21.iterrows():
        match = df_22[df_22["match_id"] == row["match_id"]].set_index("team")["goals_scored"]
        try:
            diff = match[row["team_a"]] - match[row["team_b"]]
        except KeyError:
            mismatches += 1
            continue
        mismatches += int(diff != row["goal_difference"])
    return mismatches
