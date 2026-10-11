"""Project settings and the two model specifications (2.1 and 2.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# --------------------------------------------------------------------------- method settings
RANDOM_STATE = 0          # fixed seed so every number in the report is reproducible
TEST_SIZE = 0.40          # 60% training / 40% testing (project requirement)
SKEW_THRESHOLD = 0.5      # |skewness| above this is treated as non-normal
MAX_DISCRETE_LEVELS = 5   # variables with <= 5 distinct values are not power-transformed
P_VALUE_THRESHOLD = 0.05  # backward-elimination significance level
N_REPEATED_SPLITS = 200   # robustness check: number of random 60/40 splits (lower = faster)
N_EXPLANATORY_VARIABLES = 8  # project requirement for each dataset
MAX_SHARED_VARIABLES = 4     # project requirement between 2.1 and 2.2


@dataclass(frozen=True)
class ModelConfig:
    """Everything that differs between the two regression objectives.

    Attributes:
        tag: Objective label used in file names and plot titles ("2.1" / "2.2").
        filenames: Candidate CSV file names inside the data directory (first found is used).
        target: Name of the response column.
        features: The eight explanatory variables.
        indicator_vars: Binary / ordinal variables that are never power-transformed.
        log_vars: Strictly positive variables log-transformed for skew and non-linearity.
        target_transform: None, "sqrt" or "log1p".
        labels: Human-readable names for plots.
        id_columns: Identifier columns that must never be used as predictors.
    """

    tag: str
    filenames: tuple[str, ...]
    target: str
    features: tuple[str, ...]
    indicator_vars: tuple[str, ...] = ()
    log_vars: tuple[str, ...] = ()
    target_transform: Optional[str] = None
    labels: dict = field(default_factory=dict)
    id_columns: tuple[str, ...] = ()

    def label(self, column: str) -> str:
        return self.labels.get(column, column)


OBJECTIVE_2_1 = ModelConfig(
    tag="2.1",
    filenames=("objective_2.1.csv", "objective_2_1.csv"),
    target="goal_difference",
    features=(
        "fifa_points_diff",
        "squad_value_diff_eur_m",
        "host_diff",
        "wc_appearances_diff",
        "wc_titles_diff",
        "squad_avg_age_diff",
        "rest_days_diff",
        "prior_goal_diff_diff",
    ),
    indicator_vars=("host_diff",),  # -1 / 0 / +1
    log_vars=(),                    # differences can be negative -> Yeo-Johnson instead
    target_transform=None,          # goal difference is roughly symmetric
    id_columns=("match_id", "team_a", "team_b"),
    labels={
        "goal_difference": "Goal difference (A - B)",
        "fifa_points_diff": "FIFA points diff",
        "squad_value_diff_eur_m": "Squad value diff (€m)",
        "host_diff": "Host diff",
        "wc_appearances_diff": "WC appearances diff",
        "wc_titles_diff": "WC titles diff",
        "squad_avg_age_diff": "Squad avg age diff",
        "rest_days_diff": "Rest days diff",
        "prior_goal_diff_diff": "Prior goal-diff diff",
    },
)

OBJECTIVE_2_2 = ModelConfig(
    tag="2.2",
    filenames=("objective_2.2.csv", "objective_2_2.csv"),
    target="goals_scored",
    features=(
        "team_fifa_points",
        "team_squad_value_eur_m",
        "team_is_host",
        "team_wc_appearances",
        "opponent_draw_pot",
        "team_goals_scored_before",
        "opponent_goals_conceded_before",
        "knockout_stage",
    ),
    indicator_vars=("team_is_host", "knockout_stage", "opponent_draw_pot"),
    log_vars=("team_squad_value_eur_m",),  # right-skewed, diminishing returns
    target_transform="sqrt",               # count target: right-skewed
    id_columns=("row_id", "match_id", "team", "opponent"),
    labels={
        "goals_scored": "Goals scored",
        "team_fifa_points": "Team FIFA points",
        "team_squad_value_eur_m": "Squad value (€m)",
        "team_is_host": "Team is host",
        "team_wc_appearances": "WC appearances",
        "opponent_draw_pot": "Opponent draw pot",
        "team_goals_scored_before": "Goals scored before",
        "opponent_goals_conceded_before": "Opp. goals conceded before",
        "knockout_stage": "Knockout stage",
    },
)

OBJECTIVES = {cfg.tag: cfg for cfg in (OBJECTIVE_2_1, OBJECTIVE_2_2)}

# The four concepts that appear in both models (as differences in 2.1, levels in 2.2).
SHARED_CONCEPTS = {
    "FIFA ranking points": ("fifa_points_diff", "team_fifa_points"),
    "Squad market value": ("squad_value_diff_eur_m", "team_squad_value_eur_m"),
    "Host nation": ("host_diff", "team_is_host"),
    "World Cup appearances": ("wc_appearances_diff", "team_wc_appearances"),
}
