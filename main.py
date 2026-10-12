"""Run both regression models (2.1 and 2.2).

Usage:
    python main.py

Figures are saved to outputs/figures/ and tables to outputs/results/.
"""

import logging
import sys
from pathlib import Path

# Make the package in src/ importable without installing it.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fifa_regression.config import DATA_DIR, OBJECTIVES  # noqa: E402
from fifa_regression.data import check_consistency, check_shared_variables, load_dataset  # noqa: E402
from fifa_regression.pipeline import run_analysis, save_summary  # noqa: E402


def print_headline(summary: dict) -> None:
    """Print the key results for one model."""
    final = summary["final_model"]
    row = next(r for r in summary["comparison"] if r["model"] == final)
    rob = summary["robustness"]["C: reduced" if final.startswith("C") else "B: transformed+std"]
    print(f"\n[{summary['tag']}] target = {summary['target']}")
    print(f"  final model      : {final}")
    print(f"  variables kept   : {', '.join(summary['model_c_features'])}")
    print(f"  test R2 / RMSE   : {row['R2']:.3f} / {row['RMSE']:.3f}")
    print(f"  mean test R2     : {rob['mean']:.3f} (SD {rob['sd']:.3f}) over repeated splits")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    # Project rule checks before modelling.
    check_shared_variables()
    mismatches = check_consistency(load_dataset(OBJECTIVES["2.1"], DATA_DIR),
                                   load_dataset(OBJECTIVES["2.2"], DATA_DIR))
    logging.info("2.1 vs 2.2 goal consistency check: %d mismatched matches", mismatches)

    summaries = {}
    for tag, cfg in OBJECTIVES.items():
        summaries[tag] = run_analysis(cfg)
        print_headline(summaries[tag])

    path = save_summary(summaries)
    print(f"\nDone. Figures in outputs/figures/, tables in outputs/results/ ({path.name})")


if __name__ == "__main__":
    main()
