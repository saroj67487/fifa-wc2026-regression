"""End-to-end analysis for one objective: load -> EDA -> split -> transform ->
fit -> evaluate -> diagnose -> robustness -> save outputs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

from fifa_regression import plotting
from fifa_regression.config import (
    DATA_DIR,
    N_REPEATED_SPLITS,
    OUTPUT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    ModelConfig,
)
from fifa_regression.data import describe, load_dataset, quality_report
from fifa_regression.diagnostics import fit_ols, residual_diagnostics, variance_inflation_factors
from fifa_regression.evaluation import (
    compare_models,
    regression_metrics,
    repeated_split_r2,
    summarise_robustness,
)
from fifa_regression.modeling import MODEL_A, MODEL_B, MODEL_C, fit_all_models
from fifa_regression.preprocessing import FeaturePreprocessor, target_transform

logger = logging.getLogger(__name__)


@dataclass
class OutputPaths:
    root: Path
    figures: Path = field(init=False)
    results: Path = field(init=False)

    def __post_init__(self):
        self.root = Path(self.root)
        self.figures = self.root / "figures"
        self.results = self.root / "results"
        self.figures.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)

    def result(self, cfg: ModelConfig, name: str) -> Path:
        return self.results / f"{cfg.tag}_{name}"


def run_analysis(cfg: ModelConfig, data_dir: Path = DATA_DIR, output_dir: Path = OUTPUT_DIR,
                 n_splits: int = N_REPEATED_SPLITS, seed: int = RANDOM_STATE,
                 make_plots: bool = True) -> dict:
    """Run the full pipeline for one objective and return a JSON-serialisable summary."""
    out = OutputPaths(output_dir)
    plotting.apply_style()
    logger.info("=== Linear Regression %s: predicting '%s' ===", cfg.tag, cfg.target)
    summary: dict = {"tag": cfg.tag, "target": cfg.target}

    # 1. Load and inspect -------------------------------------------------
    df = load_dataset(cfg, data_dir)
    summary["data_quality"] = quality_report(df)
    describe(df, cfg).to_csv(out.result(cfg, "descriptive_stats.csv"))

    # 2. Exploratory analysis --------------------------------------------
    corr = df[[cfg.target, *cfg.features]].corr()
    corr.to_csv(out.result(cfg, "correlation_matrix.csv"))
    if make_plots:
        plotting.target_distribution(df, cfg, out.figures)
        plotting.feature_histograms(df, cfg, out.figures)
        plotting.scatter_against_target(df, cfg, out.figures)
        plotting.correlation_heatmap(corr, cfg, out.figures)

    # 3. 60/40 split -------------------------------------------------------
    X, y = df[list(cfg.features)], df[cfg.target].astype(float)
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
    summary.update(n_train=len(X_tr_raw), n_test=len(X_te_raw))
    logger.info("Split: %d train / %d test rows", len(X_tr_raw), len(X_te_raw))

    # 4-6. Transform + standardise (fitted on training data only) ----------
    prep = FeaturePreprocessor(cfg.features, cfg.indicator_vars, cfg.log_vars).fit(X_tr_raw)
    X_tr, X_te = prep.transform(X_tr_raw), prep.transform(X_te_raw)
    X_tr_unscaled = prep.transform_unscaled(X_tr_raw)
    forward, inverse = target_transform(cfg.target_transform)
    y_tr_t = forward(y_tr)
    summary.update(log_vars=list(cfg.log_vars), yeo_johnson_vars=list(prep.yeo_johnson_vars_),
                   yj_lambdas=prep.yeo_johnson_lambdas, target_transform=cfg.target_transform)

    _transformation_table(cfg, prep, X_tr_raw, X_tr_unscaled, y_tr, y_tr_t).to_csv(
        out.result(cfg, "transformation_table.csv"), index=False)
    if make_plots:
        plotting.transformation_effect(
            _transformation_pairs(cfg, prep, X_tr_raw, X_tr_unscaled, y_tr, y_tr_t),
            cfg, out.figures)
        plotting.standardised_boxplot(X_tr, cfg, out.figures)

    # 7. Fit models --------------------------------------------------------
    models = fit_all_models(X_tr_raw, X_te_raw, X_tr, X_te, y_tr, y_tr_t, forward, inverse)
    reduced = models[MODEL_C].features
    summary["eliminated"] = models[MODEL_C].meta["eliminated"]
    summary["model_c_features"] = reduced

    # 8. Test-set evaluation ----------------------------------------------
    comparison = compare_models(models, y_tr, y_te)
    comparison.to_csv(out.result(cfg, "model_comparison.csv"), index=False)
    summary["comparison"] = comparison.to_dict(orient="records")

    # 9. Robustness: repeated 60/40 splits --------------------------------
    scores = repeated_split_r2(df, cfg, reduced, n_splits)
    scores.to_csv(out.result(cfg, "repeated_splits_r2.csv"), index=False)
    summary["robustness"] = summarise_robustness(scores)

    # Final model = transformed spec (B or C) with the higher mean test R^2 over
    # the repeated splits. Model A is a baseline only: the project requires
    # transformed and standardised inputs.
    final = MODEL_C if scores["C: reduced"].mean() >= scores["B: transformed+std"].mean() else MODEL_B
    summary["final_model"] = final
    logger.info("Final model: %s", final)

    # 10. Inference and diagnostics ---------------------------------------
    for code, feats in (("B", list(cfg.features)), ("C", reduced)):
        ols = fit_ols(X_tr[feats], y_tr_t, feats)
        ols.coefficients.to_csv(out.result(cfg, f"model{code}_coefficients.csv"))
        summary[f"model{code}_coefs"] = (ols.coefficients.reset_index()
                                        .rename(columns={"index": "term"}).to_dict("records"))
        summary[f"model{code}_fit"] = ols.fit_summary()

    vif = variance_inflation_factors(X_tr, cfg.features)
    vif.to_csv(out.result(cfg, "vif.csv"))
    summary["vif"] = vif.to_dict()

    final_feats = models[final].features
    final_ols = fit_ols(X_tr[final_feats], y_tr_t, final_feats)
    summary["diagnostics"] = residual_diagnostics(final_ols.residuals, X_tr[final_feats], final_feats)

    if make_plots:
        test_metrics = regression_metrics(y_te, models[final].test_pred, models[final].n_features)
        plotting.residual_diagnostics(final_ols.fitted, final_ols.residuals,
                                      summary["diagnostics"]["shapiro_resid_p"], final, cfg, out.figures)
        plotting.actual_vs_predicted(y_te, models[final].test_pred, test_metrics["R2"],
                                     test_metrics["RMSE"], cfg, out.figures)
        coef_b = pd.read_csv(out.result(cfg, "modelB_coefficients.csv"), index_col=0)
        plotting.standardised_coefficients(coef_b, cfg, out.figures)
        plotting.model_comparison(comparison, cfg, out.figures)
        reported = [comparison.loc[comparison.model == m, "R2"].iloc[0] for m in (MODEL_A, MODEL_B, MODEL_C)]
        plotting.repeated_splits(scores, reported, cfg, out.figures)

    return summary


def save_summary(summaries: dict, output_dir: Path = OUTPUT_DIR) -> Path:
    path = OutputPaths(output_dir).results / "summary.json"
    path.write_text(json.dumps(summaries, indent=2, default=_json_default))
    return path


# --------------------------------------------------------------------- helpers
def _transformation_table(cfg, prep, X_raw, X_unscaled, y, y_t) -> pd.DataFrame:
    rows = []
    for col in cfg.features:
        after = X_unscaled[col]
        rows.append({
            "variable": col,
            "skew_before": stats.skew(X_raw[col]),
            "skew_after": stats.skew(after),
            "shapiro_p_before": stats.shapiro(X_raw[col]).pvalue,
            "shapiro_p_after": stats.shapiro(after).pvalue,
            "transformation": prep.treatment(col),
            "r_with_target_before": np.corrcoef(X_raw[col], y)[0, 1],
            "r_with_target_after": np.corrcoef(after, y_t)[0, 1],
        })
    rows.append({
        "variable": f"{cfg.target} (target)",
        "skew_before": stats.skew(y), "skew_after": stats.skew(y_t),
        "shapiro_p_before": stats.shapiro(y).pvalue, "shapiro_p_after": stats.shapiro(y_t).pvalue,
        "transformation": cfg.target_transform or "none (approximately symmetric)",
        "r_with_target_before": np.nan, "r_with_target_after": np.nan,
    })
    return pd.DataFrame(rows)


def _transformation_pairs(cfg, prep, X_raw, X_unscaled, y, y_t):
    pairs = []
    for col in cfg.features:
        if col in cfg.log_vars:
            pairs.append((col, X_raw[col], X_unscaled[col], "log"))
        elif col in prep.yeo_johnson_vars_:
            pairs.append((col, X_raw[col], X_unscaled[col], "Yeo-Johnson"))
    if cfg.target_transform:
        pairs.append((cfg.target, y, y_t, cfg.target_transform))
    return pairs


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")
