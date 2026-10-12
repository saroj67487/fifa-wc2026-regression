"""All figures used in the report. Each function draws one figure and saves it."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: works on servers and in CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402

from fifa_regression.config import ModelConfig  # noqa: E402

# Consistent, colour-blind-safe palette across every figure.
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
LIGHT_BLUE, PALE_BLUE = "#d6e6f8", "#b9cfe9"
INK, INK2 = "#0b0b0b", "#52514e"

_BOX_STYLE = dict(
    patch_artist=True,
    boxprops=dict(facecolor=LIGHT_BLUE, color=BLUE),
    medianprops=dict(color=ORANGE),
    whiskerprops=dict(color=BLUE),
    capprops=dict(color=BLUE),
    flierprops=dict(markeredgecolor=BLUE, markersize=4),
)


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelcolor": INK2, "axes.edgecolor": "#c9c8c2", "xtick.color": INK2,
        "ytick.color": INK2, "axes.grid": True, "grid.color": "#ebeae6",
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    })


def _save(fig, out_dir: Path, name: str) -> Path:
    path = Path(out_dir) / name
    fig.savefig(path)
    plt.close(fig)
    return path


def _qq(ax, values, title: str) -> None:
    stats.probplot(values, dist="norm", plot=ax)
    ax.get_lines()[0].set(color=BLUE, markersize=4)
    ax.get_lines()[1].set(color=ORANGE)
    ax.set(title=title)


# --------------------------------------------------------------------- EDA
def target_distribution(df: pd.DataFrame, cfg: ModelConfig, out_dir: Path) -> Path:
    y = df[cfg.target]
    label = cfg.label(cfg.target)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    bins = np.arange(y.min() - 0.5, y.max() + 1.5, 1)
    ax[0].hist(y, bins=bins, color=BLUE, edgecolor="white")
    ax[0].set(title=f"Distribution of {label}", xlabel=label, ylabel="Number of rows")
    _qq(ax[1], y, f"Normal Q-Q plot of {label}")
    return _save(fig, out_dir, f"{cfg.tag}_fig01_target_distribution.png")


def feature_histograms(df: pd.DataFrame, cfg: ModelConfig, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, col in zip(axes.ravel(), cfg.features):
        ax.hist(df[col], bins=15, color=BLUE, edgecolor="white")
        ax.set(title=cfg.label(col), ylabel="Count")
        ax.text(0.97, 0.92, f"skew = {stats.skew(df[col]):.2f}", transform=ax.transAxes,
                ha="right", fontsize=9, color=INK2)
    fig.suptitle(f"{cfg.tag}: explanatory variables BEFORE transformation", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, f"{cfg.tag}_fig02_features_before.png")


def scatter_against_target(df: pd.DataFrame, cfg: ModelConfig, out_dir: Path) -> Path:
    y = df[cfg.target]
    jitter = np.random.default_rng(1).uniform(-0.12, 0.12, len(df))  # reveal overlapping points
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4))
    for ax, col in zip(axes.ravel(), cfg.features):
        ax.scatter(df[col], y + jitter, s=14, alpha=0.55, color=BLUE, edgecolor="none")
        slope, intercept = np.polyfit(df[col], y, 1)
        xs = np.linspace(df[col].min(), df[col].max(), 50)
        ax.plot(xs, slope * xs + intercept, color=ORANGE, lw=2)
        ax.set(title=f"{cfg.label(col)}  (r = {df[col].corr(y):.2f})",
               xlabel=cfg.label(col), ylabel=cfg.label(cfg.target))
    fig.suptitle(f"{cfg.tag}: each explanatory variable vs {cfg.label(cfg.target)} "
                 "(least-squares line)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, f"{cfg.tag}_fig03_scatter_vs_target.png")


def correlation_heatmap(corr: pd.DataFrame, cfg: ModelConfig, out_dir: Path) -> Path:
    names = [cfg.label(c) for c in corr.columns]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax,
                xticklabels=names, yticklabels=names,
                cbar_kws={"label": "Pearson r"}, annot_kws={"size": 8})
    ax.set_title(f"{cfg.tag}: Pearson correlation matrix")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    return _save(fig, out_dir, f"{cfg.tag}_fig04_correlation_heatmap.png")


# --------------------------------------------------------- preprocessing
def transformation_effect(pairs, cfg: ModelConfig, out_dir: Path) -> Path | None:
    """pairs: list of (column, before_values, after_values, method_name)."""
    if not pairs:
        return None
    fig, axes = plt.subplots(2, len(pairs), figsize=(3.6 * len(pairs), 6), squeeze=False)
    for j, (col, before, after, method) in enumerate(pairs):
        axes[0, j].hist(before, bins=15, color=GREY, edgecolor="white")
        axes[0, j].set(title=f"{cfg.label(col)}\nbefore (skew {stats.skew(before):.2f})")
        axes[1, j].hist(after, bins=15, color=AQUA, edgecolor="white")
        axes[1, j].set(title=f"after {method} (skew {stats.skew(after):.2f})")
    fig.suptitle(f"{cfg.tag}: effect of transformations (training data)", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, f"{cfg.tag}_fig05_transformations.png")


def standardised_boxplot(X_std: pd.DataFrame, cfg: ModelConfig, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.boxplot([X_std[c] for c in cfg.features], **_BOX_STYLE)
    ax.set_xticks(range(1, len(cfg.features) + 1), [cfg.label(c) for c in cfg.features],
                  rotation=30, ha="right")
    ax.set(title=f"{cfg.tag}: explanatory variables after standardisation (mean 0, SD 1)",
           ylabel="z-score")
    return _save(fig, out_dir, f"{cfg.tag}_fig06_standardised_boxplot.png")


# ------------------------------------------------------------ evaluation
def residual_diagnostics(fitted, residuals, shapiro_p: float, model_name: str,
                         cfg: ModelConfig, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].scatter(fitted, residuals, s=16, color=BLUE, alpha=0.7, edgecolor="none")
    axes[0].axhline(0, color=ORANGE, lw=1.5)
    axes[0].set(title="Residuals vs fitted (training)", xlabel="Fitted value", ylabel="Residual")
    _qq(axes[1], residuals, f"Normal Q-Q of residuals (Shapiro p = {shapiro_p:.3f})")
    axes[2].hist(residuals, bins=15, color=BLUE, edgecolor="white")
    axes[2].set(title="Histogram of residuals (training)", xlabel="Residual", ylabel="Count")
    fig.suptitle(f"{cfg.tag}: residual diagnostics for final model ({model_name})",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, f"{cfg.tag}_fig07_residual_diagnostics.png")


def actual_vs_predicted(y_true, y_pred, r2: float, rmse: float,
                        cfg: ModelConfig, out_dir: Path) -> Path:
    y_true = np.asarray(y_true)
    label = cfg.label(cfg.target)
    jitter = np.random.default_rng(2).uniform(-0.1, 0.1, len(y_true))
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(y_true + jitter, y_pred, s=22, color=BLUE, alpha=0.75, edgecolor="none",
               label="Test match")
    lo = min(y_true.min(), y_pred.min()) - 0.5
    hi = max(y_true.max(), y_pred.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], color=ORANGE, lw=1.8, label="Perfect prediction")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel=f"Actual {label}", ylabel=f"Predicted {label}",
           title=f"{cfg.tag}: actual vs predicted (test set)\nR² = {r2:.3f}, RMSE = {rmse:.3f}")
    ax.legend(loc="upper left", frameon=False)
    return _save(fig, out_dir, f"{cfg.tag}_fig08_actual_vs_predicted.png")


def standardised_coefficients(coefs: pd.DataFrame, cfg: ModelConfig, out_dir: Path) -> Path:
    """coefs: OLS table (with 'const' row) of the full transformed model."""
    table = coefs.drop("const")
    table = table.reindex(table["coef"].abs().sort_values().index)
    colours = [BLUE if p <= 0.05 else PALE_BLUE for p in table["p_value"]]
    unit = cfg.label(cfg.target) + (f" ({cfg.target_transform}-scale)" if cfg.target_transform else "")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.barh([cfg.label(c) for c in table.index], table["coef"], color=colours,
            xerr=[table["coef"] - table["ci_low"], table["ci_high"] - table["coef"]],
            error_kw=dict(ecolor=INK2, lw=1, capsize=3))
    ax.axvline(0, color=INK2, lw=1)
    ax.set(title=f"{cfg.tag}: standardised coefficients, Model B (95% CI)\n"
                 "dark = significant at 5%, light = not significant",
           xlabel=f"Change in {unit} per +1 SD of the variable")
    return _save(fig, out_dir, f"{cfg.tag}_fig09_standardised_coefficients.png")


def model_comparison(comparison: pd.DataFrame, cfg: ModelConfig, out_dir: Path) -> Path:
    colours = [GREY] + [BLUE] * (len(comparison) - 1)
    ticks = [m.replace(": ", ":\n") for m in comparison["model"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric in zip(axes, ["RMSE", "R2"]):
        bars = ax.bar(range(len(comparison)), comparison[metric], color=colours, edgecolor="white")
        for bar, value in zip(bars, comparison[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center",
                    va="bottom" if value >= 0 else "top", fontsize=9, color=INK)
        ax.set_xticks(range(len(comparison)), ticks, fontsize=8)
        ax.set(title=f"Test-set {'R²' if metric == 'R2' else metric}")
        ax.axhline(0, color=INK2, lw=0.8)
    fig.suptitle(f"{cfg.tag}: model comparison on the 40% test set", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir, f"{cfg.tag}_fig10_model_comparison.png")


def repeated_splits(scores: pd.DataFrame, reported_r2: list[float],
                    cfg: ModelConfig, out_dir: Path) -> Path:
    """reported_r2: test R^2 of models A, B, C on the reported split (same order as columns)."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.boxplot([scores[c] for c in scores], vert=False, **_BOX_STYLE)
    ax.set_yticks(range(1, scores.shape[1] + 1), list(scores.columns))
    for i, value in enumerate(reported_r2, start=1):
        ax.plot(value, i, marker="D", color=AQUA, markersize=7,
                label="Reported split (random_state = 0)" if i == 1 else None)
    ax.set(title=f"{cfg.tag}: test R² across {len(scores)} random 60/40 splits", xlabel="Test R²")
    ax.legend(frameon=False, loc="lower left")
    return _save(fig, out_dir, f"{cfg.tag}_fig11_repeated_splits.png")
