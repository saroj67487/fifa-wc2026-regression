# Data

Files in `raw/` are the team-compiled inputs and are never edited by the code.
All explanatory variables are known **before kick-off**; no in-match statistics are used.

## objective_2.1.csv — one row per match (104 rows)

All `*_diff` columns are team A minus team B.

| Column | Type | Description |
|---|---|---|
| match_id | int | Match identifier (not a predictor) |
| team_a, team_b | str | Team names (not predictors) |
| **goal_difference** | int | **Target:** team A goals − team B goals |
| fifa_points_diff | float | FIFA ranking points gap before the tournament |
| squad_value_diff_eur_m | float | Squad market value gap (€ million) |
| host_diff | int | +1 / 0 / −1 host-nation indicator |
| wc_appearances_diff | int | Previous World Cup appearances gap |
| wc_titles_diff | int | World Cup titles gap |
| squad_avg_age_diff | float | Squad average age gap (years) |
| rest_days_diff | int | Rest days since previous match gap |
| prior_goal_diff_diff | int | Gap in goal difference from earlier matches at this tournament |

## objective_2.2.csv — one row per team per match (208 rows)

| Column | Type | Description |
|---|---|---|
| row_id, match_id | int | Identifiers (not predictors) |
| team, opponent | str | Team names (not predictors) |
| **goals_scored** | int | **Target:** goals scored by `team` |
| team_fifa_points | float | Team's FIFA ranking points |
| team_squad_value_eur_m | float | Team's squad market value (€ million) |
| team_is_host | int | 1 if the team is a host nation |
| team_wc_appearances | int | Team's previous World Cup appearances |
| opponent_draw_pot | int | Opponent's draw pot (1 = strongest … 4 = weakest) |
| team_goals_scored_before | int | Team's goals earlier in this tournament |
| opponent_goals_conceded_before | int | Opponent's goals conceded earlier in this tournament |
| knockout_stage | int | 1 = knockout match, 0 = group match |

## Shared variables

Only four concepts appear in both datasets: FIFA ranking points, squad market value,
host status and World Cup appearances (max allowed: 4). `tests/test_pipeline.py` enforces this.

## Integrity checks (run automatically)

- No missing values or duplicate rows.
- For all 104 matches, `goal_difference` in 2.1 equals the two `goals_scored` values in 2.2.
- Knockout goals are recorded after extra time, excluding penalty shoot-outs.

## Sources

- FIFA: https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/statistics
- The Stats Don't Lie: https://www.thestatsdontlie.com/football/world-cup-2026/
- FBref: https://fbref.com/en/
