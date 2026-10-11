# FIFA World Cup 2026 – Linear Regression (Assessment 3, Objective 2)

Two linear regression models built on the 104 matches of the FIFA World Cup 2026:

| Model | Predicts | Rows | Final variables | Mean test R² (200 splits) |
|---|---|---|---|---|
| **2.1** | Goal difference between two teams | 104 matches | FIFA points gap, squad value gap, WC titles gap, squad age gap | 0.337 |
| **2.2** | Goals scored by one team | 208 team-match rows | Squad value (log), opponent draw pot | 0.145 |

Both models use the same pipeline: exploratory analysis → normality check → log / Yeo–Johnson
transformation → z-score standardisation → **60% / 40% train–test split** → OLS regression →
backward elimination → test-set evaluation → assumption diagnostics → robustness check over
200 random splits.

**Team:** Saroj Poudel (s396872) · Rajib Sadaula (s399118) · Raju Chaulagain (s398426) ·
Saad Mahmud Osmani (s400601) — Charles Darwin University

---

## Quick start

```bash
git clone <your-repo-url>
cd fifa-wc2026-regression

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py                     # runs 2.1 and 2.2, writes outputs/
```

Run the tests (from the repo root):

```bash
# macOS / Linux
PYTHONPATH=src python -m unittest discover -s tests -t .
# Windows (PowerShell)
$env:PYTHONPATH="src"; python -m unittest discover -s tests -t .
```

The same commands, with Windows variants, are at the top of `requirements.txt`.
Settings such as the split ratio, seed and number of robustness splits are in `src/fifa_regression/config.py`.

## Project structure

```
fifa-wc2026-regression/
├── data/
│   ├── raw/                     # input datasets (never modified)
│   │   ├── objective_2.1.csv    # 104 matches, 8 explanatory variables
│   │   └── objective_2.2.csv    # 208 team-match rows, 8 explanatory variables
│   └── README.md                # data dictionary and sources
├── main.py                      # run everything: python main.py
├── src/fifa_regression/
│   ├── config.py                # constants + the two model specifications
│   ├── data.py                  # loading, validation, descriptive statistics
│   ├── preprocessing.py         # log / Yeo-Johnson transforms + standardisation
│   ├── modeling.py              # null, raw (A), full (B) and reduced (C) models
│   ├── diagnostics.py           # OLS inference, VIF, Breusch-Pagan, Durbin-Watson
│   ├── evaluation.py            # metrics + repeated-split robustness check
│   ├── plotting.py              # the 11 figures per model used in the report
│   └── pipeline.py              # end-to-end run for one objective
├── tests/                       # unit tests + integration tests on the real data
├── outputs/
│   ├── figures/                 # 22 PNG figures (generated)
│   └── results/                 # CSV tables + summary.json (generated)
├── docs/methodology.md          # method decisions and why
├── reports/                     # final report (PDF / Word export)
├── requirements.txt             # packages + step-by-step run commands
└── pyproject.toml
```

## How the pipeline works

| Step | Module | What happens |
|---|---|---|
| 1. Load & validate | `data.py` | Checks columns, 8 predictors, no ID leakage, no missing values |
| 2. Explore | `pipeline.py`, `plotting.py` | Histograms, Q–Q plot, scatter plots, correlation heatmap |
| 3. Split | `pipeline.py` | `train_test_split(test_size=0.40, random_state=0)` |
| 4. Transform | `preprocessing.py` | Log for squad value (2.2); Yeo–Johnson for continuous vars with \|skew\| > 0.5; √ target for 2.2 |
| 5. Standardise | `preprocessing.py` | z-score every predictor (fitted on training data only) |
| 6. Fit | `modeling.py` | Null, A (raw), B (all 8 transformed), C (backward elimination, p ≤ 0.05) |
| 7. Evaluate | `evaluation.py` | MAE, MSE, RMSE, NRMSE, R², adjusted R² on the 40% test set |
| 8. Diagnose | `diagnostics.py` | t-tests, 95% CIs, F-test, VIF, Shapiro–Wilk, Breusch–Pagan, Durbin–Watson |
| 9. Robustness | `evaluation.py` | Full pipeline repeated over 200 random 60/40 splits |

Transformation and scaling parameters are always learned from the **training split only**,
so nothing from the test set leaks into the model. See [docs/methodology.md](docs/methodology.md)
for the reasoning behind each decision.

## Results

| | 2.1 Goal difference | 2.2 Goals scored |
|---|---|---|
| Training R² / adjusted R² | 0.557 / 0.526 | 0.298 / 0.286 (√goals scale) |
| Test R² (reported split) | 0.127 | 0.121 |
| Test RMSE / MAE | 2.047 / 1.551 | 1.383 / 0.997 |
| Null-model test RMSE | 2.208 | 1.476 |
| Mean test R² over 200 splits | 0.337 (SD 0.119) | 0.145 (SD 0.089) |

Every number above is in `outputs/results/summary.json`, and re-running the code reproduces it exactly.

## Requirements checklist

- [x] 2.1 dataset: 104 rows, exactly 8 pre-match explanatory variables
- [x] 2.2 dataset: 208 rows, exactly 8 pre-match explanatory variables
- [x] At most 4 shared variables (FIFA points, squad value, host, WC appearances) — enforced in tests
- [x] Variables standardised to a common scale
- [x] Non-normal variables transformed; non-linear relationship (squad value) linearised
- [x] 60% training / 40% testing split
- [x] Labelled figures with interpretation (see report)

## Data sources

[FIFA official statistics](https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/statistics) ·
[The Stats Don't Lie](https://www.thestatsdontlie.com/football/world-cup-2026/) ·
[FBref](https://fbref.com/en/) — see [data/README.md](data/README.md).
