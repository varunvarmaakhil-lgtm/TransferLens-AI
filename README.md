# TransferLens AI

AI-assisted football recruitment for Europe's top leagues. Ask in plain English who could replace a striker, whether a player is undervalued, or why he might move. Every number comes from a validated model or a database query; a small local language model only chooses tools and explains their results.

## Results

| Component | How it was tested | Result |
| --- | --- | --- |
| Transfer prediction (will a player change club next season?) | Season by season, each test season unseen in training | ROC-AUC 0.73 (2019-2024), 0.75 on 2022-2024 for the live model; logistic regression 0.68 |
| Fair-price ranges (80% interval for market value) | Season by season, conformal calibration | 79.8% of real values inside the range; median error 19.8% vs 31.7% for last season's value |
| Similar-player search | A player's previous-season profile must find himself among ~366 same-position players | Top 10 in 34.1% of cases vs 3.3% by chance |
| Explanations (SHAP) | Contributions must sum to each prediction | Max error 1.8e-14 |
| Agent (local Qwen3 8B, 7 tools) | Manual question set, grounding check on every answer | Comparisons and rankings computed in code; untraced numbers sent back for correction |

## The honest-evaluation story

The first version reported 86% accuracy and 0.93 ROC-AUC. An audit found two problems:

1. **Random split.** 84% of test rows belonged to players already in training (0.92 random vs 0.81 grouped by player).
2. **Label drift.** Transfer records are four times sparser in early seasons (4,055 in 2012/13 vs 17,379 in 2023/24), so the label rose from 7% to 51% for reasons that had nothing to do with football.

The label was rebuilt from 1.9M match appearances (a change of main club between seasons), every model is now tested season by season, and the honest score rose from 0.64 to 0.73 with 17 time-safe club and playing-time features. Full details: EVALUATION.md.

## Example

    you> Compare Archie Gray and Warren Zaire-Emery: who is the better value signing right now?
      -> compare_players(names=['Archie Gray', 'Warren Zaire-Emery'])
    Better value (largest gap between fair value and market value): Warren Zaire-Emery (Archie Gray 20.3M gap, Warren Zaire-Emery 44.2M gap).
    Cheaper: Archie Gray (35.0M vs 60.0M).
    [1 tool call: compare_players | grounding: 10/10 numbers traced to tool outputs]

## Architecture

    Transfermarkt datalake (DuckDB: appearances, valuations, games, clubs)
      -> build_label.py, features_v2.py, build_player_seasons.py   one time-safe player-season table
      -> transfer_now.py   CatBoost transfer model
      -> fair_price.py     CatBoost 10/50/90 quantiles + conformal calibration
      -> similar.py        weighted profile search within sub-position
      -> explain.py        SHAP drivers for every player
      -> tools.py          7 tools with JSON schemas, validated in code
      -> agent.py          Qwen3 8B via Ollama, tool loop, grounding self-check

## Design rules for the language model

- The model never calculates, compares or ranks; tools return verdicts computed in code.
- Every tool argument is validated (types, ranges, known names and positions) before it runs.
- Every number in an answer is checked against tool outputs; untraced numbers go back to the model once.

## Run it

    brew install ollama && brew services start ollama && ollama pull qwen3:8b
    python agent.py "Find 5 centre-backs under 23 who are undervalued"

## Roadmap

1. Honest evaluation and transfer model (done)
2. Fair-price ranges, undervalued detection, similarity search (done)
3. Local agent with validated tools (done)
4. SHAP explanations (done)
5. Squad and budget optimiser (integer programming with PuLP)
6. 50-question agent test set, Streamlit app, demo

## Limitations

- Market values are Transfermarkt estimates, not transfer fees; the fair-price model leans on reputation (last value, club squad value).
- No contract, wage or injury history, so availability is only partly captured.
- About a third of player-seasons have no next-season appearances (retired or left the covered leagues) and are excluded from the transfer label.

## Data

Nodeball Football Datalake by salimt (github.com/salimt), built on Transfermarkt. See README_datalake.md.

## Squad and budget optimiser (Phase 5)

Given a club and a budget, TransferLens finds the club's weak positions (reliable players per position against a target depth, plus players likely to leave and ageing players). It then chooses signings with an integer program (scipy milp / HiGHS), not a greedy heuristic.

Constraints: the budget at market value, the number of signings, per-position caps and minimums, at most one player per selling club, and an optional age limit.

Evaluation: 40 clubs x budgets of 20, 50 and 100 million EUR x 2 objectives, with greedy given the same rules.

| Result | Value |
|---|---|
| Valid scenarios | 228 |
| Optimiser worse than greedy | never |
| Quality objective: strictly better than greedy | 95% of scenarios, median +35.6% total fair value |
| Bargains objective: strictly better than greedy | 100% of scenarios |
| Greedy failed to build a valid plan | 10 of 228 |

Example: FC Barcelona, 80 million EUR, 3 signings: Kerkez (LB), Vicario (GK), Doan (RW). 80.0M spent for 117.3M of fair value, solved in 0.01s.

The agent reaches the optimiser through two tools, squad_needs and plan_signings. It presents the optimiser's picks as returned and asks for a budget if none is given.

Known limitation: availability (move probability) is a filter, not part of the objective, so a plan can include players who are unlikely to be sold.
