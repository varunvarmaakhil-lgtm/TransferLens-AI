"""TransferLens squad needs and signing optimiser (integer programming, HiGHS via SciPy).

Usage:
  python optimiser.py needs "FC Barcelona"
  python optimiser.py plan "FC Barcelona" --budget 80 --max-signings 3
  python optimiser.py eval
"""
import argparse
import difflib
import time
import unicodedata
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds

D = "/Users/varunvarma/Transferlens/data/processed/"
SEASON = 2025
TARGET_DEPTH = {"Goalkeeper": 2, "Centre-Back": 4, "Left-Back": 2, "Right-Back": 2, "Defensive Midfield": 2,
                "Central Midfield": 3, "Attacking Midfield": 2, "Left Winger": 2, "Right Winger": 2,
                "Centre-Forward": 2}
RELIABLE = {"min_minutes_share": 0.2, "max_move_probability": 0.5, "max_age": 32}
_cache = {}


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


def table():
    if "t" not in _cache:
        b = pd.read_csv(D + "player_seasons.csv", low_memory=False)
        b = b[b["season"] == SEASON].copy()
        fp = pd.read_csv(D + "fair_price_%d.csv" % SEASON, usecols=["player_id", "fair_low", "fair_mid", "fair_high"])
        tp = pd.read_csv(D + "transfer_prob_%d.csv" % SEASON)
        b = b.merge(fp, on="player_id", how="left").merge(tp, on="player_id", how="left")
        b["club_key"] = b["club_name"].map(_norm)
        _cache["t"] = b.reset_index(drop=True)
    return _cache["t"]


CLUB_ALIASES = {"barca": "barcelona", "psg": "paris saint", "man utd": "manchester united",
                "man united": "manchester united", "man city": "manchester city", "spurs": "tottenham",
                "juve": "juventus", "atleti": "atletico", "bayern": "bayern", "bvb": "dortmund",
                "real": "real madrid", "inter milan": "internazionale", "ac milan": "associazione calcio milan"}


def resolve_club(name):
    t, key = table(), _norm(name)
    key = CLUB_ALIASES.get(key, key)
    sizes = t.groupby("club_name")["minutes_all"].sum()
    names = sizes.index.tolist()
    exact = [c for c in names if _norm(c) == key]
    if exact:
        return exact[0], None
    contains = [c for c in names if key in _norm(c)]
    if contains:
        return max(contains, key=lambda c: sizes[c]), None
    close = difflib.get_close_matches(key, [_norm(c) for c in names], n=5, cutoff=0.6)
    return None, {"error": "club_not_found", "query": name,
                  "did_you_mean": [c for c in names if _norm(c) in close]}


def squad_needs(club):
    club_name, err = resolve_club(club)
    if err:
        return err
    t = table()
    sq = t[t["club_name"] == club_name].copy()
    sq["reliable"] = ((sq["minutes_share"] >= RELIABLE["min_minutes_share"])
                      & (sq["p_move"].fillna(0) < RELIABLE["max_move_probability"])
                      & (sq["age"] < RELIABLE["max_age"]))
    rows = []
    for pos, target in TARGET_DEPTH.items():
        p = sq[sq["sub_position"] == pos]
        reliable = int(p["reliable"].sum())
        risk = p[(p["minutes_share"] >= RELIABLE["min_minutes_share"])
                 & (p["p_move"].fillna(0) >= RELIABLE["max_move_probability"])]
        old = p[(p["minutes_share"] >= RELIABLE["min_minutes_share"]) & (p["age"] >= RELIABLE["max_age"])]
        rows.append({"position": pos, "target_depth": target, "squad_players": len(p),
                     "reliable_players": reliable, "need": max(0, target - reliable),
                     "likely_to_leave": ["%s (%d%%)" % (r["name"], round(100 * r["p_move"])) for _, r in risk.iterrows()],
                     "ageing": ["%s (%d)" % (r["name"], r["age"]) for _, r in old.iterrows()]})
    rows.sort(key=lambda r: (-r["need"], r["reliable_players"]))
    return {"club": club_name, "season": "%d/%d" % (SEASON, SEASON + 1),
            "rule": "reliable = at least %d%% of club minutes, move probability under %d%%, age under %d"
                    % (100 * RELIABLE["min_minutes_share"], 100 * RELIABLE["max_move_probability"], RELIABLE["max_age"]),
            "priority_positions": [r["position"] for r in rows if r["need"] > 0],
            "positions": rows}


def candidates(club_name, positions, budget_m, max_age=None, min_minutes_share=0.4,
               min_move_probability=0.0, per_position=60, objective="quality"):
    t = table()
    c = t[(t["club_name"] != club_name) & (t["sub_position"].isin(positions))
          & (t["minutes_share"] >= min_minutes_share) & t["mv_end"].notna() & t["fair_mid"].notna()
          & (t["mv_end"] >= 0.5e6) & (t["mv_end"] <= budget_m * 1e6)
          & (t["p_move"].fillna(0) >= min_move_probability)].copy()
    if max_age is not None:
        c = c[c["age"] <= max_age]
    c["cost_m"] = c["mv_end"] / 1e6
    c["fair_m"] = c["fair_mid"] / 1e6
    c["score"] = c["fair_m"] if objective == "quality" else (c["fair_m"] - c["cost_m"])
    c = c[c["score"] > 0]
    c = c.sort_values("score", ascending=False).groupby("sub_position").head(per_position)
    return c.reset_index(drop=True)


def solve(c, budget_m, max_signings, caps, minimums, club_limit=1):
    n = len(c)
    if n == 0:
        return None
    A, lo, hi = [c["cost_m"].values, np.ones(n)], [0, 0], [budget_m, max_signings]
    for pos, cap in caps.items():
        row = (c["sub_position"] == pos).values.astype(float)
        A.append(row)
        lo.append(minimums.get(pos, 0))
        hi.append(cap)
    for club, idx in c.groupby("club_name").groups.items():
        if len(idx) > club_limit:
            row = np.zeros(n)
            row[list(idx)] = 1
            A.append(row)
            lo.append(0)
            hi.append(club_limit)
    res = milp(c=-c["score"].values, constraints=LinearConstraint(np.vstack(A), lo, hi),
               integrality=np.ones(n), bounds=Bounds(0, 1), options={"time_limit": 20})
    if res.x is None or res.status not in (0, 1):
        return None
    return np.round(res.x).astype(bool)


def greedy(c, budget_m, max_signings, caps, minimums=None, club_limit=1):
    minimums = minimums or {}
    ratio = (c["score"] / c["cost_m"].clip(lower=0.1)).sort_values(ascending=False)
    pick, state = np.zeros(len(c), bool), {"spent": 0.0, "used": {}, "clubs": {}}

    def fits(r):
        return (pick.sum() < max_signings and state["spent"] + r["cost_m"] <= budget_m
                and state["used"].get(r["sub_position"], 0) < caps.get(r["sub_position"], 0)
                and state["clubs"].get(r["club_name"], 0) < club_limit)

    def take(i, r):
        pick[i] = True
        state["spent"] += r["cost_m"]
        state["used"][r["sub_position"]] = state["used"].get(r["sub_position"], 0) + 1
        state["clubs"][r["club_name"]] = state["clubs"].get(r["club_name"], 0) + 1

    for pos, need in minimums.items():
        for i in ratio.index:
            if state["used"].get(pos, 0) >= need:
                break
            r = c.loc[i]
            if r["sub_position"] == pos and not pick[i] and fits(r):
                take(i, r)
    if any(state["used"].get(p, 0) < n for p, n in minimums.items()):
        return None
    for i in ratio.index:
        r = c.loc[i]
        if not pick[i] and fits(r):
            take(i, r)
    return pick


def plan_signings(club, budget_eur_m, max_signings=3, positions=None, max_age=None, objective="quality",
                  min_move_probability=0.0, compare_greedy=True):
    needs = squad_needs(club)
    if "error" in needs:
        return needs
    club_name = needs["club"]
    need_map = {r["position"]: r["need"] for r in needs["positions"]}
    if positions:
        caps = {p: max(1, need_map.get(p, 1)) for p in positions}
    else:
        caps = {p: n for p, n in need_map.items() if n > 0}
    if not caps:
        return {"club": club_name, "error": "no_needs",
                "detail": "every position meets its target depth; pass positions to plan anyway"}
    c = candidates(club_name, list(caps), budget_eur_m, max_age, min_move_probability=min_move_probability,
                   objective=objective)
    if c.empty:
        return {"club": club_name, "error": "no_candidates", "detail": "no players match the budget and filters"}
    top_needs = [p for p in needs["priority_positions"] if p in caps][:max_signings]
    mins = {p: 1 for p in top_needs}
    t0 = time.time()
    x, relaxed = solve(c, budget_eur_m, max_signings, caps, mins), False
    if x is None:
        mins = {}
        x, relaxed = solve(c, budget_eur_m, max_signings, caps, mins), True
    secs = time.time() - t0
    if x is None or not x.any():
        return {"club": club_name, "error": "no_feasible_plan", "detail": "budget too small for any candidate"}
    pick = c[x]
    out = {"club": club_name, "budget_eur_m": budget_eur_m, "objective": objective,
           "objective_meaning": "maximise total model fair value" if objective == "quality"
           else "maximise total gap between fair value and market value",
           "positions_targeted": caps,
           "signings": [{"name": r["name"], "from_club": r["club_name"], "position": r["sub_position"],
                         "age": int(round(r["age"])), "market_value_eur_m": round(r["cost_m"], 1),
                         "fair_mid_eur_m": round(r["fair_m"], 1),
                         "move_probability_pct": None if pd.isna(r["p_move"]) else int(round(100 * r["p_move"])),
                         "undervalued": bool(r["mv_end"] < r["fair_low"])} for _, r in pick.iterrows()],
           "total_cost_eur_m": round(pick["cost_m"].sum(), 1),
           "budget_left_eur_m": round(budget_eur_m - pick["cost_m"].sum(), 1),
           "total_fair_value_eur_m": round(pick["fair_m"].sum(), 1),
           "objective_value": round(pick["score"].sum(), 1),
           "constraints": ["budget %.0f%% used" % (100 * pick["cost_m"].sum() / budget_eur_m),
                           "%d of %d signings" % (len(pick), max_signings),
                           "at most one player from each selling club"]
                          + (["could not fill every top need within budget, so minimums were relaxed"] if relaxed else []),
           "solver": "integer program (HiGHS via SciPy), %d candidates, %.2fs" % (len(c), secs)}
    if compare_greedy:
        g = greedy(c, budget_eur_m, max_signings, caps, mins)
        out["greedy_baseline_objective_value"] = None if g is None else round(c.loc[g, "score"].sum(), 1)
        out["greedy_baseline_note"] = ("same rules; greedy could not find a valid plan" if g is None
                                       else "same rules; best value per euro first")
    return out


def evaluate(n_clubs=40, budgets=(20, 50, 100), max_signings=3, seed=42):
    t = table()
    sizes = t.groupby("club_name")["minutes_all"].sum().sort_values(ascending=False)
    clubs = sizes.index[:200].to_series().sample(n=min(n_clubs, 200), random_state=seed).tolist()
    rows = []
    for club in clubs:
        for b in budgets:
            for obj in ("quality", "bargains"):
                p = plan_signings(club, b, max_signings, objective=obj)
                if "error" in p:
                    continue
                rows.append({"club": club, "budget": b, "objective": obj, "optimiser": p["objective_value"],
                             "greedy": np.nan if p["greedy_baseline_objective_value"] is None
                             else p["greedy_baseline_objective_value"],
                             "budget_used": p["total_cost_eur_m"] / b, "n_signings": len(p["signings"])})
    r = pd.DataFrame(rows)
    r["gain_pct"] = 100 * (r["optimiser"] - r["greedy"]) / r["greedy"].replace(0, np.nan)
    r.to_csv("results/optimiser_eval.csv", index=False)
    print("scenarios: %d (clubs x budgets x objectives) | greedy found no valid plan in %d"
          % (len(r), int(r["greedy"].isna().sum())))
    r = r.dropna(subset=["greedy"])
    for obj, s in r.groupby("objective"):
        print("  %-8s optimiser >= greedy in %.0f%% | strictly better in %.0f%% | median gain %.1f%% | mean gain %.1f%%"
              % (obj, 100 * (s.optimiser >= s.greedy - 1e-6).mean(), 100 * (s.optimiser > s.greedy + 0.05).mean(),
                 s.gain_pct.median(), s.gain_pct.mean()))
    print("  never worse than greedy: %s" % bool((r.optimiser >= r.greedy - 1e-6).all()))
    print("saved results/optimiser_eval.csv")
    return r


if __name__ == "__main__":
    import json
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["needs", "plan", "eval"])
    ap.add_argument("club", nargs="?")
    ap.add_argument("--budget", type=float, default=50)
    ap.add_argument("--max-signings", type=int, default=3)
    ap.add_argument("--max-age", type=float)
    ap.add_argument("--objective", default="quality", choices=["quality", "bargains"])
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    if a.cmd == "eval":
        evaluate()
    elif a.cmd == "needs":
        print(json.dumps(squad_needs(a.club), ensure_ascii=False, indent=1))
    else:
        print(json.dumps(plan_signings(a.club, a.budget, a.max_signings, max_age=a.max_age, objective=a.objective),
                         ensure_ascii=False, indent=1))
