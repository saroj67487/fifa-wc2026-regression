# Methodology decisions

Short record of each modelling choice and the reason for it. Settings live in
`src/fifa_regression/config.py`.

## Train–test split
- 60% / 40% with `random_state = 0` (project requirement; fixed seed for reproducibility).
- Every fitted parameter (Yeo–Johnson λ, scaler mean/SD, regression coefficients) is learned
  on the training rows only.

## Normality and transformations
- Skewness and Shapiro–Wilk are computed for every variable.
- |skewness| > 0.5 ⇒ transform:
  - **Yeo–Johnson** for continuous variables that can be zero or negative (differences, goal counts).
  - **Natural log** for `team_squad_value_eur_m` (2.2): strictly positive, right-skewed, and its
    scatter plot shows diminishing returns — the log both normalises and linearises it.
- **Not transformed:** binary / ordinal indicators and any variable with ≤ 5 distinct values
  (`host_diff`, `rest_days_diff`, `team_is_host`, `opponent_draw_pot`, `knockout_stage`).
  A monotone transform cannot make a few-level variable normal, and doing so produced unstable
  λ estimates on some splits.
- **Targets:** 2.1 goal difference is roughly symmetric (training skew 0.10) and left as is.
  2.2 goals scored is a right-skewed count (skew 0.97) and modelled on the √ scale; predictions
  are squared back to goals before any metric is computed.

## Standardisation
- z-score for every predictor, so coefficients are "change in target per +1 SD" and directly
  comparable. Standardisation does not change OLS predictions, only interpretation.

## Models
| Name | Inputs | Purpose |
|---|---|---|
| Null | training mean | Benchmark every model must beat |
| A | 8 raw variables | Baseline showing the effect of pre-processing |
| B | 8 transformed + standardised | Full model |
| C | B after backward elimination (p > 0.05 removed one at a time) | Simpler, final model |

## Choosing the final model
A single 40% test set (42 rows in 2.1) is noisy, so the choice between B and C uses the mean
test R² over 200 random 60/40 splits, with the full pipeline re-fitted each time.
Model A is not eligible because the brief requires transformed and standardised inputs.

## Diagnostics
- t-tests and 95% CIs per coefficient; overall F-test.
- VIF (multicollinearity, threshold 5).
- Shapiro–Wilk on residuals, Q–Q plot (normality).
- Breusch–Pagan (constant variance).
- Durbin–Watson (independence, ~2 is ideal).

Implemented in `diagnostics.py` with NumPy/SciPy, using the same textbook formulas as `statsmodels.OLS().summary()`.

## Known limitations
- Small samples; test metrics vary a lot between splits.
- 2.2 has two rows per match, so rows from one match can fall in both train and test sets.
- Goals are counts — Poisson regression would be a natural extension.
- "Goals before" variables are cumulative and grow with matches played.
