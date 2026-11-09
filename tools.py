"""TransferLens tool layer.

Every number the agent reports comes from one of these functions. Each tool has a
JSON schema (sent to the language model) and is validated in code before it runs.
All tables are precomputed, so every call is a fast lookup.
"""
import contextlib
import difflib
import io
import json
import re
import unicodedata
import numpy as np
import pandas as pd

D = "/Users/varunvarma/Transferlens/data/processed/"
SEASON = 2025
_cache = {}


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


def _table():
    if "t" not in _cache:
        b = pd.read_csv(D + "player_seasons.csv", low_memory=False)
        b = b[b["season"] == SEASON].copy()
        fp = pd.read_csv(D + "fair_price_%d.csv" % SEASON,
                         usecols=["player_id", "fair_low", "fair_mid", "fair_high"])
        tp = pd.read_csv(D + "transfer_prob_%d.csv" % SEASON)
        b = b.merge(fp, on="player_id", how="left").merge(tp, on="player_id", how="left")
        b["key"] = b["name"].map(_norm)
        _cache["t"] = b.reset_index(drop=True)
    return _cache["t"]


def _m(x):
    return None if pd.isna(x) else round(float(x) / 1e6, 1)


def _r(x, n=2):
    return None if pd.isna(x) else round(float(x), n)


def _resolve(name):
    t, key = _table(), _norm(name)
    hit = t[t["key"] == key]
    if hit.empty:
        hit = t[t["key"].str.contains(key, regex=False)]
    if hit.empty and len(key.split()) > 1:
        sur = key.split()[-1]
        by_sur = t[t["key"].str.split().str[-1] == sur]
        if by_sur["key"].nunique() == 1:
            hit = by_sur
    if hit.empty:
        keys = t["key"].tolist()
        close = set(difflib.get_close_matches(key, keys, n=5, cutoff=0.6))
        last = {k.split()[-1]: k for k in keys if k}
        close |= {last[m] for m in difflib.get_close_matches(key.split()[-1], list(last), n=5, cutoff=0.75)}
        names = t[t["key"].isin(close)]["name"].head(5).tolist()
        return None, {"error": "player_not_found", "query": name, "did_you_mean": names}
    if len(hit) > 1 and hit["key"].nunique() > 1:
        hit = hit.sort_values("minutes_all", ascending=False)
        if len(hit) > 1 and hit.iloc[0]["key"] != key:
            return None, {"error": "ambiguous_name", "query": name,
                          "candidates": [{"name": r["name"], "club": r["club_name"]} for _, r in hit.head(5).iterrows()]}
    return hit.sort_values("minutes_all", ascending=False).iloc[0], None


def _transfer_quality():
    try:
        with open("/Users/varunvarma/Transferlens/models/transfer_now.json") as fh:
            auc = json.load(fh)["eval_2022_2024"]["roc_auc_mean"]
        return "ROC-AUC %.2f on unseen seasons 2022-2024; treat as an estimate" % auc
    except (OSError, KeyError, ValueError):
        return "ROC-AUC about 0.73 on unseen seasons; treat as an estimate"


def _undervalued(r):
    return bool(pd.notna(r["fair_low"]) and pd.notna(r["mv_end"]) and r["mv_end"] < r["fair_low"])


def find_player(name):
    r, err = _resolve(name)
    if err:
        return err
    return {"name": r["name"], "club": r["club_name"], "position": r["position"],
            "sub_position": r["sub_position"], "age": _r(r["age"], 1), "season": "2025/26",
            "minutes": int(r["minutes_all"]) if pd.notna(r["minutes_all"]) else None,
            "minutes_share": _r(r["minutes_share"]), "goals": int(r["goals"]), "assists": int(r["assists"]),
            "goals_plus_assists_per90": _r(r["ga_p90"]), "market_value_eur_m": _m(r["mv_end"])}


def transfer_probability(name):
    r, err = _resolve(name)
    if err:
        return err
    base = float(_table()["p_move"].mean())
    return {"name": r["name"], "club": r["club_name"], "season": "2025/26",
            "probability_changes_club_pct": _r(100 * r["p_move"], 0),
            "average_probability_all_players_pct": _r(100 * base, 0),
            "context": {"minutes_share_pct": _r(100 * r["minutes_share"], 0), "club_tenure_seasons": _r(r["club_tenure"], 0),
                        "club_final_position": _r(r["club_final_position"], 0), "club_win_rate": _r(r["club_win_rate"]),
                        "age": _r(r["age"], 1)},
            "model_quality": _transfer_quality()}


def fair_price(name):
    r, err = _resolve(name)
    if err:
        return err
    mv, lo, mid, hi = _m(r["mv_end"]), _m(r["fair_low"]), _m(r["fair_mid"]), _m(r["fair_high"])
    if mid is None or mv is None:
        verdict = "no fair value available"
    elif _undervalued(r):
        verdict = "undervalued: market value is below the whole fair range"
    elif hi is not None and mv > hi:
        verdict = "overvalued: market value is above the whole fair range"
    else:
        verdict = "fairly valued: market value is inside the fair range"
    return {"name": r["name"], "club": r["club_name"], "fair_value_eur_m": mid, "market_value_eur_m": mv,
            "fair_range_eur_m": {"low": lo, "mid": mid, "high": hi},
            "summary_use_this": "%s: fair value EUR %sM (80%% range %s-%sM), market value EUR %sM; %s."
                                % (r["name"], mid, lo, hi, mv, verdict),
            "undervalued": _undervalued(r),
            "model_quality": "80% range; real values fell inside it 79.8% of the time in testing"}


def similar_players(name, k=5, max_age=None, max_value_eur_m=None):
    import similar as sim
    if _sub(name):
        return {"error": "looks_like_a_position", "query": name,
                "hint": "similar_players needs a player name; use search_players(sub_position='%s') instead" % _sub(name)}
    r, err = _resolve(name)
    if err:
        return err
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            out = sim.similar(r["name"], k=k, max_age=max_age, max_value=max_value_eur_m, season=SEASON)
    except SystemExit:
        return {"error": "not_enough_minutes", "query": name,
                "detail": "similarity needs at least 900 minutes this season"}
    return {"reference": {"name": r["name"], "club": r["club_name"], "sub_position": r["sub_position"]},
            "filters": {"max_age": max_age, "max_value_eur_m": max_value_eur_m},
            "players": [{"name": p["name"], "club": p["club_name"], "age": _r(p["age"], 0),
                         "goals_plus_assists_per90": _r(p["ga_p90"]), "minutes_share": _r(p["minutes_share"]),
                         "market_value_eur_m": _m(p["mv_end"]), "fair_mid_eur_m": _m(p["fair_mid"]),
                         "undervalued": p["undervalued"] == "yes"} for _, p in out.iterrows()]}


SUB_POSITIONS = ["Goalkeeper", "Centre-Back", "Left-Back", "Right-Back", "Defensive Midfield",
                 "Central Midfield", "Attacking Midfield", "Left Midfield", "Right Midfield",
                 "Left Winger", "Right Winger", "Second Striker", "Centre-Forward"]
ALIASES = {"gk": "Goalkeeper", "keeper": "Goalkeeper", "cb": "Centre-Back", "center back": "Centre-Back",
           "centre back": "Centre-Back", "central defender": "Centre-Back", "lb": "Left-Back",
           "left back": "Left-Back", "rb": "Right-Back", "right back": "Right-Back",
           "dm": "Defensive Midfield", "cdm": "Defensive Midfield", "holding midfielder": "Defensive Midfield",
           "cm": "Central Midfield", "central midfielder": "Central Midfield", "am": "Attacking Midfield",
           "cam": "Attacking Midfield", "number 10": "Attacking Midfield", "lw": "Left Winger",
           "left winger": "Left Winger", "rw": "Right Winger", "right winger": "Right Winger",
           "st": "Centre-Forward", "cf": "Centre-Forward", "striker": "Centre-Forward",
           "centre forward": "Centre-Forward", "center forward": "Centre-Forward", "9": "Centre-Forward"}
SORTS = ["value_gap", "goals_plus_assists_per90", "minutes_share", "youngest", "cheapest"]


def _sub(v):
    k = _norm(v).replace("-", " ").rstrip("s")
    if k in ALIASES:
        return ALIASES[k]
    for s in SUB_POSITIONS:
        if _norm(s).replace("-", " ").rstrip("s") == k:
            return s
    m = difflib.get_close_matches(k, [_norm(s).replace("-", " ") for s in SUB_POSITIONS], n=1, cutoff=0.7)
    return SUB_POSITIONS[[_norm(s).replace("-", " ") for s in SUB_POSITIONS].index(m[0])] if m else None


def search_players(sub_position=None, max_age=None, max_value_eur_m=None, min_value_eur_m=1.0,
                   min_minutes_share=0.4, undervalued_only=False, sort_by="value_gap", k=5, exclude_club=None):
    t = _table().copy()
    s = t[(t["minutes_share"] >= min_minutes_share) & (t["mv_end"] >= min_value_eur_m * 1e6)]
    if exclude_club:
        s = s[~s["club_name"].fillna("").map(_norm).str.contains(_norm(exclude_club), regex=False)]
    if sub_position:
        s = s[s["sub_position"] == sub_position]
    if max_age is not None:
        s = s[s["age"] <= max_age]
    if max_value_eur_m is not None:
        s = s[s["mv_end"] <= max_value_eur_m * 1e6]
    s = s.assign(gap=s["fair_mid"] - s["mv_end"], undervalued=(s["mv_end"] < s["fair_low"]))
    if undervalued_only:
        s = s[s["undervalued"]]
    key = {"value_gap": ("gap", False), "goals_plus_assists_per90": ("ga_p90", False),
           "minutes_share": ("minutes_share", False), "youngest": ("age", True),
           "cheapest": ("mv_end", True)}[sort_by]
    s = s.sort_values(key[0], ascending=key[1], na_position="last").head(k)
    summary = {}
    if len(s):
        top = s.iloc[0]
        mv = s.loc[s["p_move"].idxmax()] if s["p_move"].notna().any() else None
        ch = s.loc[s["mv_end"].idxmin()]
        summary = {"first_in_ranking": "%s (%s)" % (top["name"], top["club_name"]),
                   "most_likely_to_move": None if mv is None else "%s, %d%%" % (mv["name"], round(100 * mv["p_move"])),
                   "cheapest": "%s, EUR %.1fM" % (ch["name"], ch["mv_end"] / 1e6)}
    return {"filters": {"sub_position": sub_position, "max_age": max_age, "max_value_eur_m": max_value_eur_m,
                        "min_value_eur_m": min_value_eur_m, "min_minutes_share": min_minutes_share,
                        "undervalued_only": undervalued_only, "sort_by": sort_by, "exclude_club": exclude_club},
            "summary_computed_in_code": summary,
            "count_shown": len(s),
            "players": [{"name": p["name"], "club": p["club_name"], "sub_position": p["sub_position"],
                         "age": _r(p["age"], 0), "minutes_share_pct": _r(100 * p["minutes_share"], 0),
                         "goals_plus_assists_per90": _r(p["ga_p90"]), "market_value_eur_m": _m(p["mv_end"]),
                         "fair_mid_eur_m": _m(p["fair_mid"]), "undervalued": bool(p["undervalued"]),
                         "move_probability_pct": _r(100 * p["p_move"], 0)} for _, p in s.iterrows()]}


def compare_players(names):
    rows, errors = [], []
    for n in names:
        r, err = _resolve(n)
        (errors.append(err) if err else rows.append(r))
    if errors:
        return {"error": "player_not_found", "details": errors}
    if len(rows) < 2:
        return {"error": "need_two_players"}
    out = [{"name": r["name"], "club": r["club_name"], "sub_position": r["sub_position"], "age": _r(r["age"], 1),
            "goals": int(r["goals"]), "assists": int(r["assists"]),
            "goals_plus_assists_per90": _r(r["ga_p90"]), "minutes_share_pct": _r(100 * r["minutes_share"], 0),
            "market_value_eur_m": _m(r["mv_end"]), "fair_mid_eur_m": _m(r["fair_mid"]),
            "value_gap_eur_m": _m(r["fair_mid"] - r["mv_end"]), "undervalued": _undervalued(r),
            "move_probability_pct": _r(100 * r["p_move"], 0)} for r in rows]

    def best(field, lowest=False):
        vals = [(o[field], o["name"]) for o in out if o[field] is not None]
        if not vals:
            return None
        v = (min if lowest else max)(vals)[0]
        tied = [n for x, n in vals if x == v]
        return tied[0] if len(tied) == 1 else "tie: " + ", ".join(tied)
    verdict = {"cheaper_market_value": best("market_value_eur_m", lowest=True),
               "more_goals_plus_assists_per90": best("goals_plus_assists_per90"),
               "more_goals_plus_assists_total": None,
               "plays_more": best("minutes_share_pct"),
               "better_value_largest_gap_fair_minus_market": best("value_gap_eur_m"),
               "younger": best("age", lowest=True),
               "more_likely_to_move": best("move_probability_pct")}
    tot = [(o["goals"] + o["assists"], o["name"]) for o in out]
    m = max(tot)[0]
    tied = [n for x, n in tot if x == m]
    verdict["more_goals_plus_assists_total"] = tied[0] if len(tied) == 1 else "tie: " + ", ".join(tied)
    def line(label, who, field, unit):
        if not who or who.startswith("tie"):
            return "%s: %s" % (label, who or "unknown")
        vals = ", ".join("%s %s%s" % (o["name"], o[field], unit) for o in out)
        return "%s: %s (%s)" % (label, who, vals)
    sentences = [
        line("Better value (largest gap between fair value and market value)",
             verdict["better_value_largest_gap_fair_minus_market"], "value_gap_eur_m", "M gap"),
        line("Cheaper", verdict["cheaper_market_value"], "market_value_eur_m", "M"),
        line("More goals plus assists per 90", verdict["more_goals_plus_assists_per90"], "goals_plus_assists_per90", ""),
        line("Plays more", verdict["plays_more"], "minutes_share_pct", "%"),
        line("More likely to move", verdict["more_likely_to_move"], "move_probability_pct", "%")]
    return {"players": out, "comparison_computed_in_code": verdict, "verdicts_use_these": sentences}


LABELS = {"minutes_share": "share of club minutes", "apps_share": "share of club appearances",
          "minutes_share_change": "change in minutes share vs last season", "club_tenure": "seasons at current club",
          "new_arrival": "joined the club this season", "squad_size": "squad size",
          "minutes_rank_pct": "minutes rank in squad (0 = most)", "club_turnover": "club squad turnover",
          "mv_end": "market value", "mv_log_change": "market value change vs last season",
          "mv_vs_peak": "market value vs career peak", "mv_vs_club_median": "market value vs squad median",
          "club_median_mv": "club squad median value", "mv_rank_in_club_pct": "value rank in squad (0 = highest)",
          "club_win_rate": "club win rate", "club_managers": "managers at club this season",
          "club_final_position": "club league finish", "age": "age", "goals": "goals", "assists": "assists",
          "goals_p90": "goals per 90", "assists_p90": "assists per 90", "ga_p90": "goals plus assists per 90",
          "minutes_all": "minutes played", "apps_all": "appearances", "height_in_cm": "height (cm)",
          "main_comp": "league", "sub_position": "position", "position": "position group",
          "country_of_citizenship": "nationality", "foot": "preferred foot",
          "prev_moved_rate": "share of past seasons with a club change", "prev_moves": "number of past club changes",
          "mv_end_prev1": "market value last season", "mv_end_prev2": "market value two seasons ago",
          "mv_peak_prev": "career peak market value before this season", "ga_p90_prev1": "goals plus assists per 90 last season",
          "minutes_share_prev1": "minutes share last season", "minutes_all_prev1": "minutes played last season",
          "seasons_in_data": "seasons in the data", "yellow": "yellow cards", "red": "red cards"}


LEAGUES = {
    "GB1": "Premier League", "ES1": "LaLiga", "IT1": "Serie A", "L1": "Bundesliga", "FR1": "Ligue 1",
    "NL1": "Eredivisie", "PO1": "Liga Portugal", "BE1": "Belgian Pro League", "TR1": "Super Lig (Turkey)",
    "SC1": "Scottish Premiership", "RU1": "Russian Premier League", "UKR1": "Ukrainian Premier League",
    "GR1": "Greek Super League", "DK1": "Danish Superliga", "A1": "Austrian Bundesliga", "C1": "Swiss Super League",
    "GB2": "Championship", "ES2": "LaLiga 2", "IT2": "Serie B", "L2": "2. Bundesliga", "FR2": "Ligue 2",
    "CL": "Champions League", "EL": "Europa League", "UCOL": "Conference League",
    "CLQ": "Champions League qualifying", "ELQ": "Europa League qualifying", "ECLQ": "Conference League qualifying",
    "USC": "UEFA Super Cup", "KLUB": "Club World Cup",
    "FAC": "FA Cup", "CGB": "EFL Cup", "GBCS": "Community Shield", "CDR": "Copa del Rey", "SUC": "Supercopa",
    "CIT": "Coppa Italia", "SCI": "Supercoppa Italiana", "DFB": "DFB-Pokal", "DFL": "DFL-Supercup",
    "FRCH": "Trophee des Champions", "CDF": "Coupe de France", "NLP": "KNVB Cup", "NLSC": "Johan Cruijff Schaal",
    "POCP": "Taca de Portugal", "POSU": "Supertaca", "BESC": "Belgian Super Cup", "SFA": "Scottish Cup",
    "RUP": "Russian Cup", "RUSS": "Russian Super Cup", "UKRP": "Ukrainian Cup", "GRP": "Greek Cup",
    "DKP": "Danish Cup", "TRP": "Turkish Cup", "TRS": "Turkish Super Cup",
    "FIWC": "World Cup (national team)", "EURO": "EURO (national team)", "EMQ": "EURO qualifying (national team)",
    "WMQ": "World Cup qualifying (national team)", "UNL": "Nations League (national team)",
    "COPA": "Copa America (national team)", "AFCN": "Africa Cup of Nations (national team)",
}


def league_name(code):
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return "-"
    return LEAGUES.get(str(code), str(code))


def _fmt_value(feature, v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "unknown"
    if feature == "main_comp":
        return league_name(v)
    if isinstance(v, str):
        try:
            v = float(v)
        except ValueError:
            return "unknown" if v in ("missing", "nan", "None") else v
        if pd.isna(v):
            return "unknown"
    if ("share" in feature and "rank" not in feature) or feature in ("club_win_rate", "club_turnover", "prev_moved_rate"):
        return "%.0f%%" % (100 * v)
    if feature.startswith("mv_end") or feature in ("club_median_mv", "mv_peak_prev"):
        return "EUR %.1fM" % (v / 1e6)
    if feature in ("new_arrival",):
        return "yes" if v >= 0.5 else "no"
    return "%g" % round(v, 2)


def explain_player(name, k=5):
    r, err = _resolve(name)
    if err:
        return err
    if "ex" not in _cache:
        _cache["ex"] = pd.read_csv(D + "explanations_%d.csv" % SEASON)
    ex = _cache["ex"][_cache["ex"]["player_id"] == r["player_id"]]
    out = {"name": r["name"], "club": r["club_name"],
           "move_probability_pct": _r(100 * r["p_move"], 0), "fair_mid_eur_m": _m(r["fair_mid"]),
           "method": "SHAP contributions from the CatBoost models; drivers ranked by size; direction computed in code"}
    for model, key, up, down in [("transfer", "why_move_probability", "raises move probability", "lowers move probability"),
                                 ("fairprice", "why_fair_value", "raises fair value", "lowers fair value")]:
        e = ex[ex["model"] == model].sort_values("rank").head(k)
        out[key] = ["%s = %s: %s (%s)" % (LABELS.get(x.feature, x.feature.replace("_", " ")),
                                           _fmt_value(x.feature, x.value if isinstance(x.value, str) else
                                                      (None if pd.isna(x.value) else float(x.value))),
                                           up if x.shap > 0 else down, x.strength) for x in e.itertuples()]
    return out


def _all_seasons():
    import os
    if "all" not in _cache:
        path = D + "player_seasons_all.csv"
        _cache["all"] = pd.read_csv(path if os.path.exists(path) else D + "player_seasons.csv", low_memory=False)
    return _cache["all"]


def outside_players(since=2018, min_peak_eur_m=5):
    """Known players who are not in this season's covered leagues (moved to MLS, Saudi Arabia, Brazil, ...,
    or retired): last seen in the data since 2018 and a career peak value of at least EUR 5M."""
    a, t = _all_seasons(), _table()
    last = a.sort_values("season").groupby("player_id").tail(1)
    peak = a.groupby("player_id")["mv_end"].max()
    cur = last[(last["season"] >= since) & ~last["player_id"].isin(t["player_id"])].copy()
    cur["peak"] = cur["player_id"].map(peak)
    cur = cur[cur["peak"].fillna(0) >= min_peak_eur_m * 1e6]
    return cur.sort_values("peak", ascending=False)["name"].drop_duplicates().tolist()


def career(name):
    """Season-by-season history for one player, from the full table (including seasons outside the covered
    leagues, which are marked)."""
    a = _all_seasons()
    r, err = _resolve(name)
    if err:
        hit = a[a["name"].map(_norm) == _norm(name)]
        if hit.empty:
            return err
        r = hit.sort_values("season").iloc[-1]
    h = a[a["player_id"] == r["player_id"]].sort_values("season")
    rows = []
    for _, s in h.iterrows():
        covered = pd.notna(s["club_name"]) and s.get("squad_size", 0) >= 11
        rows.append({"season": "%d/%02d" % (s["season"], (s["season"] + 1) % 100),
                     "club": s["club_name"] if covered else "outside covered leagues",
                     "league": league_name(s["main_comp"]) if covered else "-",
                     "age": _r(s["age"], 0), "apps": int(s["apps_all"]) if pd.notna(s["apps_all"]) else 0,
                     "minutes": int(s["minutes_all"]) if pd.notna(s["minutes_all"]) else 0,
                     "goals": int(s["goals"]) if pd.notna(s["goals"]) else 0,
                     "assists": int(s["assists"]) if pd.notna(s["assists"]) else 0,
                     "goals_plus_assists_per90": _r(s["ga_p90"]),
                     "market_value_eur_m": _m(s["mv_end"]), "covered": bool(covered)})
    cov = [x for x in rows if x["covered"]]
    clubs = list(dict.fromkeys(x["club"] for x in cov))
    return {"name": r["name"], "seasons": rows,
            "totals_covered_seasons": {"seasons": len(cov), "clubs": clubs,
                                       "apps": sum(x["apps"] for x in cov), "minutes": sum(x["minutes"] for x in cov),
                                       "goals": sum(x["goals"] for x in cov), "assists": sum(x["assists"] for x in cov),
                                       "peak_market_value_eur_m": max([x["market_value_eur_m"] or 0 for x in rows] or [0])},
            "note": "Seasons 2012/13 onwards, all competitions in the data (mainly European leagues, cups and "
                    "national-team games). Earlier seasons and uncovered leagues are not counted."}


def squad_needs(club):
    import optimiser
    return optimiser.squad_needs(club)


def plan_signings(club, budget_eur_m, max_signings=3, positions=None, max_age=None, objective="quality"):
    import optimiser
    return optimiser.plan_signings(club, budget_eur_m, max_signings, positions=positions, max_age=max_age,
                                   objective=objective)


def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


NAME = {"type": "string", "description": "Player name, e.g. 'Viktor Gyokeres'"}
SCHEMAS = [
    _fn("find_player", "Profile of one player this season: club, position, age, minutes, goals, assists, market value.",
        {"name": NAME}, ["name"]),
    _fn("compare_players", "Compare 2-4 named players side by side. Returns which one is cheaper, more productive, "
        "better value and more likely to move, computed in code. Use this for any comparison.",
        {"names": {"type": "array", "items": {"type": "string"}, "description": "2-4 player names"}}, ["names"]),
    _fn("explain_player", "Why the models rate a player as they do: top drivers of his move probability and of "
        "his fair value, each with direction and strength. Use for any why question.",
        {"name": NAME, "k": {"type": "integer", "description": "Drivers per model, 1-8, default 5"}}, ["name"]),
    _fn("transfer_probability", "Model probability (percent) that one player changes club after this season.",
        {"name": NAME}, ["name"]),
    _fn("fair_price", "Model fair market value range (EUR millions) for one player and whether he is undervalued.",
        {"name": NAME}, ["name"]),
    _fn("similar_players", "Players with a similar profile to a named reference player, same position, optionally "
        "younger or cheaper. Needs a real player name.",
        {"name": NAME, "k": {"type": "integer", "description": "How many players, 1-15, default 5"},
         "max_age": {"type": "number", "description": "Maximum age"},
         "max_value_eur_m": {"type": "number", "description": "Maximum market value in EUR millions"}}, ["name"]),
    _fn("squad_needs", "A club's squad needs this season: reliable players per position against a target depth, "
        "players likely to leave, ageing players, and priority positions.",
        {"club": {"type": "string", "description": "Club name, e.g. 'FC Barcelona'"}}, ["club"]),
    _fn("plan_signings", "Best set of signings for a club within a budget, chosen by an integer-programming optimiser "
        "that fills the club's priority positions. Use for any 'who should club X sign with budget Y' question.",
        {"club": {"type": "string", "description": "Club name"},
         "budget_eur_m": {"type": "number", "description": "Transfer budget in EUR millions (market values)"},
         "max_signings": {"type": "integer", "description": "Maximum number of signings, 1-5, default 3"},
         "positions": {"type": "array", "items": {"type": "string", "enum": SUB_POSITIONS},
                       "description": "Only plan for these positions (optional; default = the club's needs)"},
         "max_age": {"type": "number", "description": "Maximum age (optional)"},
         "objective": {"type": "string", "enum": ["quality", "bargains"],
                       "description": "quality = most total fair value; bargains = biggest gap between fair and market value"}},
        ["club", "budget_eur_m"]),
    _fn("search_players", "Search all players by position and filters, e.g. undervalued centre-backs under 26. "
        "Use this when the user names a position or role rather than a reference player. Results include move probability.",
        {"sub_position": {"type": "string", "enum": SUB_POSITIONS},
         "max_age": {"type": "number", "description": "Maximum age"},
         "max_value_eur_m": {"type": "number", "description": "Maximum market value in EUR millions"},
         "min_value_eur_m": {"type": "number", "description": "Minimum market value in EUR millions, default 1"},
         "min_minutes_share": {"type": "number", "description": "Minimum share of club minutes 0-1, default 0.4"},
         "undervalued_only": {"type": "boolean", "description": "Only players valued below their fair range"},
         "sort_by": {"type": "string", "enum": SORTS, "description": "Default value_gap"},
         "k": {"type": "integer", "description": "How many players, 1-15, default 5"},
         "exclude_club": {"type": "string", "description": "Leave out players already at this club, e.g. the user's own club"}}, []),
]
TOOLS = {"find_player": find_player, "explain_player": explain_player, "compare_players": compare_players, "transfer_probability": transfer_probability,
         "fair_price": fair_price, "similar_players": similar_players, "search_players": search_players,
         "squad_needs": squad_needs, "plan_signings": plan_signings}
LIMITS = {"k": (1, 15), "max_age": (15, 45), "max_value_eur_m": (0, 500), "min_value_eur_m": (0, 500),
          "min_minutes_share": (0, 1), "budget_eur_m": (1, 1000), "max_signings": (1, 5)}


def validate(name, args):
    if name not in TOOLS:
        return None, {"error": "unknown_tool", "tool": name, "available": list(TOOLS)}
    schema = next(s["function"]["parameters"] for s in SCHEMAS if s["function"]["name"] == name)
    args = dict(args or {})
    for k in set(args) - set(schema["properties"]):
        args.pop(k)
    clean = {}
    for k, v in args.items():
        if v is None or v == "":
            continue
        typ = schema["properties"][k]["type"]
        try:
            if typ == "array":
                if isinstance(v, str):
                    v = [x for x in re.split(r",| and | vs\.? | versus ", v) if x.strip()]
                v = [str(x).strip() for x in v if str(x).strip()]
                if k == "positions":
                    mapped = [_sub(x) for x in v]
                    if not v or None in mapped or len(v) > 5:
                        return None, {"error": "bad_argument", "argument": k, "value": v, "allowed": SUB_POSITIONS}
                    clean[k] = list(dict.fromkeys(mapped))
                    continue
                if not 2 <= len(v) <= 4:
                    return None, {"error": "bad_argument", "argument": k, "value": v, "expected": "2-4 names"}
                clean[k] = v
                continue
            if typ == "boolean":
                clean[k] = v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes")
                continue
            if k == "sub_position":
                sp = _sub(v)
                if not sp:
                    return None, {"error": "bad_argument", "argument": k, "value": v, "allowed": SUB_POSITIONS}
                clean[k] = sp
                continue
            if typ == "integer":
                v = int(float(v))
            elif typ == "number":
                v = float(v)
            else:
                v = str(v).strip()
        except (TypeError, ValueError):
            return None, {"error": "bad_argument", "argument": k, "value": v, "expected": typ}
        if k in LIMITS and not LIMITS[k][0] <= v <= LIMITS[k][1]:
            return None, {"error": "out_of_range", "argument": k, "value": v, "allowed": LIMITS[k]}
        if "enum" in schema["properties"][k]:
            match = [e for e in schema["properties"][k]["enum"] if e.lower().startswith(str(v).lower()[:3])]
            if not match:
                return None, {"error": "bad_argument", "argument": k, "value": v,
                              "allowed": schema["properties"][k]["enum"]}
            v = match[0]
        clean[k] = v
    missing = [k for k in schema["required"] if k not in clean]
    if missing:
        return None, {"error": "missing_argument", "missing": missing}
    return clean, None


def call(name, args):
    clean, err = validate(name, args)
    if err:
        return err
    try:
        return TOOLS[name](**clean)
    except Exception as e:
        return {"error": "tool_failed", "tool": name, "detail": str(e)[:200]}


if __name__ == "__main__":
    for q in [("compare_players", {"names": ["Gyokeres", "Pavlidis"]}),
              ("search_players", {"sub_position": "CB", "max_age": 26, "undervalued_only": True, "k": 5}),
              ("similar_players", {"name": "Centre-back"})]:
        print(">>>", q[0], q[1])
        print(json.dumps(call(*q), ensure_ascii=False, indent=1)[:900], "\n")
