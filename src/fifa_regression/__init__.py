"""Linear regression analysis of FIFA World Cup 2026 matches (Assessment 3, Objective 2).

Two models share one pipeline:
    * 2.1 - goal difference between two teams in a match (104 matches)
    * 2.2 - goals scored by one team in a match (208 team-match rows)

Typical use::

    from fifa_regression import run_analysis, OBJECTIVE_2_1
    result = run_analysis(OBJECTIVE_2_1)
"""

from fifa_regression.config import OBJECTIVE_2_1, OBJECTIVE_2_2, OBJECTIVES, ModelConfig
from fifa_regression.pipeline import run_analysis

__all__ = ["ModelConfig", "OBJECTIVE_2_1", "OBJECTIVE_2_2", "OBJECTIVES", "run_analysis"]
__version__ = "1.0.0"
