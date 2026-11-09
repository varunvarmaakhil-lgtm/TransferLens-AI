"""Known-answer evaluation of the TransferLens agent.

Each case states which tool must be called, which arguments it must get, and which facts must
appear in the answer. The facts are computed by calling the tools directly, so they follow the data.

  python agent_eval.py --check          ground truth only, no language model (seconds)
  python agent_eval.py                  full run, about 12-15 minutes
  python agent_eval.py --only 34 42     rerun some cases
  python agent_eval.py --model qwen3:4b
"""
import argparse
import os
import re
import time
import pandas as pd
import tools
import agent

N = tools._norm


def norm_text(s):
    return re.sub(r"[-\s]+", " ", N(s))


def name_tokens(s):
    stop = {"fc", "cf", "ac", "sc", "afc", "club", "de", "the", "real", "sv", "1.", "cp"}
    toks = [t for t in norm_text(s).split() if t not in stop and len(t) >= 3]
    return toks or [norm_text(s)]


def C(cid, cat, q, tools_ok, args=None, truth=None, forbid=(), must=(), note=""):
    return {"id": cid, "cat": cat, "q": q, "tools": tools_ok, "args": args or {}, "truth": truth,
            "forbid": set(forbid), "must": must, "note": note}


# truth helpers: each returns a list of facts that must appear in the answer
def f_num(fn, path, tol=None, *a):
    def run():
        r = getattr(tools, fn)(*a)
        if "error" in r:
            raise ValueError(r)
        v = r
        for p in path:
            v = v[p]
        return [("num", v, tol)]
    return run


def f_name(fn, pick, *a, kind="name"):
    def run():
        r = getattr(tools, fn)(*a)
        if "error" in r:
            raise ValueError(r)
        vals = pick(r)
        vals = vals if isinstance(vals, list) else [vals]
        return [(kind, v, None) for v in vals if v and not str(v).startswith("tie")]
    return run


def verdict(key):
    return lambda r: r["comparison_computed_in_code"][key]


def surnames(r):
    return [s["name"] for s in r["signings"]]


CB, LB, GK, CF, DM, RW, AM = "Centre-Back", "Left-Back", "Goalkeeper", "Centre-Forward", \
    "Defensive Midfield", "Right Winger", "Attacking Midfield"
has = lambda word: (lambda v: word in N(v))
le = lambda x: (lambda v: v is not None and float(v) <= x + 1e-9)
eq = lambda x: (lambda v: v == x)

CASES = [
    # lookups
    C(1, "lookup", "Which club does Viktor Gyokeres play for?", {"find_player"},
      truth=f_name("find_player", lambda r: r["club"], "Viktor Gyokeres", kind="name")),
    C(2, "lookup", "How old is Lamine Yamal?", {"find_player"},
      truth=f_num("find_player", ["age"], 1.0, "Lamine Yamal")),
    C(3, "lookup", "What is Bukayo Saka's market value?", {"find_player", "fair_price"},
      truth=f_num("find_player", ["market_value_eur_m"], None, "Bukayo Saka")),
    C(4, "lookup", "How many goals has Vangelis Pavlidis scored this season?", {"find_player", "compare_players"},
      truth=f_num("find_player", ["goals"], None, "Vangelis Pavlidis")),
    C(5, "lookup", "Tell me about Joao Neves.", {"find_player", "fair_price", "transfer_probability", "explain_player"},
      truth=f_name("find_player", lambda r: r["club"], "Joao Neves")),
    # transfer probability
    C(6, "probability", "How likely is Pedri to leave his club?", {"transfer_probability", "explain_player"},
      truth=f_num("transfer_probability", ["probability_changes_club_pct"], 0.6, "Pedri")),
    C(7, "probability", "What are the chances Milos Kerkez moves this season?", {"transfer_probability", "explain_player"},
      truth=f_num("transfer_probability", ["probability_changes_club_pct"], 0.6, "Milos Kerkez")),
    C(8, "probability", "Will Guglielmo Vicario change clubs?", {"transfer_probability", "explain_player"},
      truth=f_num("transfer_probability", ["probability_changes_club_pct"], 0.6, "Guglielmo Vicario")),
    C(9, "probability", "Transfer probability for Ritsu Doan", {"transfer_probability"},
      truth=f_num("transfer_probability", ["probability_changes_club_pct"], 0.6, "Ritsu Doan")),
    C(10, "probability", "Is Robert Lewandowski likely to move?", {"transfer_probability", "explain_player"},
      truth=f_num("transfer_probability", ["probability_changes_club_pct"], 0.6, "Robert Lewandowski")),
    # fair price
    C(11, "fair_price", "What is Vangelis Pavlidis really worth?", {"fair_price"},
      truth=f_num("fair_price", ["fair_range_eur_m", "mid"], None, "Vangelis Pavlidis")),
    C(12, "fair_price", "Is Bukayo Saka overvalued or undervalued?", {"fair_price", "compare_players"},
      truth=f_num("fair_price", ["market_value_eur_m"], None, "Bukayo Saka")),
    C(13, "fair_price", "Fair price for Samu Aghehowa", {"fair_price"},
      truth=f_num("fair_price", ["fair_range_eur_m", "mid"], None, "Samu Aghehowa")),
    C(14, "fair_price", "How much is Declan Rice worth according to the model?", {"fair_price"},
      truth=f_num("fair_price", ["fair_range_eur_m", "mid"], None, "Declan Rice")),
    C(15, "fair_price", "What is the fair value range for Milos Kerkez?", {"fair_price"},
      truth=lambda: f_num("fair_price", ["fair_range_eur_m", "low"], None, "Milos Kerkez")()
      + f_num("fair_price", ["fair_range_eur_m", "high"], None, "Milos Kerkez")()),
    # compare
    C(16, "compare", "Gyokeres or Pavlidis, who is better value?", {"compare_players"}, forbid={"fair_price"},
      truth=f_name("compare_players", verdict("better_value_largest_gap_fair_minus_market"), ["Gyokeres", "Pavlidis"])),
    C(17, "compare", "Compare Archie Gray and Warren Zaire-Emery", {"compare_players"},
      truth=f_name("compare_players", verdict("better_value_largest_gap_fair_minus_market"),
                   ["Archie Gray", "Warren Zaire-Emery"])),
    C(18, "compare", "Who is cheaper, Guglielmo Vicario or David Raya?", {"compare_players"},
      truth=f_name("compare_players", verdict("cheaper_market_value"), ["Guglielmo Vicario", "David Raya"])),
    C(19, "compare", "Saka vs Lamine Yamal: who is more likely to move?", {"compare_players"},
      truth=f_name("compare_players", verdict("more_likely_to_move"), ["Bukayo Saka", "Lamine Yamal"])),
    C(20, "compare", "Compare Pedri, Vitinha and Joao Neves on value", {"compare_players"},
      args={"names": lambda v: len(v) == 3},
      truth=f_name("compare_players", verdict("better_value_largest_gap_fair_minus_market"),
                   ["Pedri", "Vitinha", "Joao Neves"])),
    # explain
    C(21, "explain", "Why might Vangelis Pavlidis move this season?", {"explain_player"}),
    C(22, "explain", "Why is Milos Kerkez valued so highly?", {"explain_player"}),
    C(23, "explain", "Explain Robert Lewandowski's transfer probability", {"explain_player"}),
    C(24, "explain", "What drives Samu Aghehowa's fair value?", {"explain_player"}),
    # similar players
    C(25, "similar", "Find me a cheaper Gyokeres under 25", {"similar_players"}, args={"max_age": le(25)}),
    C(26, "similar", "Players similar to Vangelis Pavlidis", {"similar_players"}, args={"name": has("pavlidis")}),
    C(27, "similar", "Who plays like Lamine Yamal but costs under 30 million?", {"similar_players"},
      args={"max_value_eur_m": le(30)}),
    C(28, "similar", "Find 3 players like Milos Kerkez", {"similar_players"}, args={"k": eq(3)}),
    C(29, "similar", "Alternatives to Declan Rice aged 27 or younger", {"similar_players"}, args={"max_age": le(27)}),
    # search
    C(30, "search", "Best undervalued centre-backs aged 23 or under", {"search_players"},
      args={"sub_position": eq(CB), "max_age": le(23)}),
    C(31, "search", "Left-backs under 25 worth less than 20 million", {"search_players"},
      args={"sub_position": eq(LB), "max_age": le(25), "max_value_eur_m": le(20)}),
    C(32, "search", "Top strikers by goals plus assists per 90", {"search_players"},
      args={"sub_position": eq(CF), "sort_by": eq("goals_plus_assists_per90")}),
    C(33, "search", "Youngest regular goalkeepers", {"search_players"},
      args={"sub_position": eq(GK), "sort_by": eq("youngest")}),
    C(34, "search", "I am Barcelona's sporting director. Find us good centre-backs.", {"search_players"},
      args={"sub_position": eq(CB), "exclude_club": has("barcelona")}, note="known failure: exclude_club"),
    C(35, "search", "We are Arsenal. Find a defensive midfielder aged 26 or under.", {"search_players"},
      args={"sub_position": eq(DM), "max_age": le(26), "exclude_club": has("arsenal")}),
    C(36, "search", "Cheapest right wingers who play regularly", {"search_players"},
      args={"sub_position": eq(RW), "sort_by": eq("cheapest")}),
    C(37, "search", "Find 8 attacking midfielders aged 21 or under", {"search_players"},
      args={"sub_position": eq(AM), "max_age": le(21), "k": eq(8)}),
    # squad needs
    C(38, "needs", "What does Barcelona's squad need?", {"squad_needs"}, args={"club": has("barcelona")},
      truth=f_name("squad_needs", lambda r: r["priority_positions"][0], "Barcelona", kind="pos")),
    C(39, "needs", "Where are Arsenal weakest in their squad?", {"squad_needs"}, args={"club": has("arsenal")},
      truth=f_name("squad_needs", lambda r: r["priority_positions"][0], "Arsenal", kind="pos")),
    C(40, "needs", "What are Real Madrid's squad gaps?", {"squad_needs"}, args={"club": has("madrid")},
      truth=f_name("squad_needs", lambda r: r["priority_positions"][0], "Real Madrid", kind="pos")),
    # plan signings
    C(41, "plan", "Barcelona have 80 million euros. Who should they sign?", {"plan_signings"},
      args={"budget_eur_m": eq(80.0)}, truth=f_name("plan_signings", surnames, "Barcelona", 80)),
    C(42, "plan", "Arsenal want a striker and a winger, max 26 years old, budget 60 million. Plan it.",
      {"plan_signings"}, args={"budget_eur_m": eq(60.0), "max_age": le(26), "max_signings": eq(2)},
      note="known failure: 3 signings for 2 players"),
    C(43, "plan", "Liverpool have 50 million. Find them the best bargains.", {"plan_signings"},
      args={"budget_eur_m": eq(50.0), "objective": eq("bargains")}),
    C(44, "plan", "Juventus want two signings with 40 million euros.", {"plan_signings"},
      args={"budget_eur_m": eq(40.0), "max_signings": eq(2)}),
    C(45, "plan", "PSG have 100 million for a centre-back and a goalkeeper. Plan it.", {"plan_signings"},
      args={"budget_eur_m": eq(100.0), "positions": lambda v: set(v) == {CB, GK}}),
    C(46, "plan", "Plan Benfica's window with 30 million, players aged 24 or under.", {"plan_signings"},
      args={"budget_eur_m": eq(30.0), "max_age": le(24)}),
    # missing budget: must ask, must not plan
    C(47, "ask_budget", "Who should Benfica sign?", None, forbid={"plan_signings"}, must=("budget",)),
    C(48, "ask_budget", "Plan Chelsea's next transfer window.", None, forbid={"plan_signings"}, must=("budget",)),
    # robustness
    C(49, "robust", "How likely is Zxqvor Blimpton to move?", {"transfer_probability", "find_player"}),
    C(50, "robust", "Lewandowsky transfer chances?", {"transfer_probability", "explain_player"},
      truth=f_num("transfer_probability", ["probability_changes_club_pct"], 0.6, "Robert Lewandowski")),
    C(51, "off_topic", "What is the capital of France?", set()),
    C(52, "off_topic", "Write me a short poem about the weather.", set()),
]


def fact_ok(fact, answer):
    kind, v, tol = fact
    if kind == "num":
        tol = tol if tol is not None else max(0.01 * abs(v), 0.051)
        return any(abs(n - v) <= tol for n in agent.numbers(answer))
    if kind == "pos":
        return norm_text(v) in norm_text(answer)
    words = set(re.findall(r"[a-z0-9]+", norm_text(answer)))
    return any(tok in words for tok in name_tokens(v))


def score(case, res, facts):
    trace, answer = res["trace"], res["answer"]
    called = [t["tool"] for t in trace]
    why = []
    if case["tools"] is None:
        route = True
    elif not case["tools"]:
        route = not called
    else:
        route = any(t in case["tools"] for t in called)
    executed = {t["tool"] for t in trace if not t.get("error")}   # a call refused by code did not run
    route = route and not (case["forbid"] & executed)
    if not route:
        why.append("route: called %s" % (called or "nothing"))
    args_ok = True
    if case["args"]:
        cleaned = [tools.validate(t["tool"], t["args"])[0] or {} for t in trace if t["tool"] in (case["tools"] or ())]

        def fits(a):
            return all(k in a and chk(a[k]) for k, chk in case["args"].items())
        args_ok = any(fits(a) for a in cleaned)
        if not args_ok:
            why.append("args: %s" % (cleaned or "none"))
    missing = [f for f in facts if not fact_ok(f, answer)]
    facts_ok = not missing
    if missing:
        why.append("missing facts: %s" % [f[1] for f in missing])
    must_ok = all(w in answer.lower() for w in case["must"])
    if not must_ok:
        why.append("answer lacks %s" % list(case["must"]))
    g = res.get("grounding")
    grounded = g is not None and g[0] == g[1]
    if not grounded:
        why.append("grounding %s" % (g,))
    return {"route": route, "args": args_ok, "facts": facts_ok and must_ok, "grounded": grounded,
            "pass": route and args_ok and facts_ok and must_ok and grounded, "why": "; ".join(why)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", nargs="*", type=int)
    ap.add_argument("--model", default="qwen3:8b")
    a = ap.parse_args()
    cases = [c for c in CASES if not a.only or c["id"] in a.only]
    truth, bad = {}, 0
    for c in cases:
        try:
            truth[c["id"]] = c["truth"]() if c["truth"] else []
        except Exception as e:
            bad += 1
            print("  TRUTH FAILED case %d (%s): %s" % (c["id"], c["q"], str(e)[:200]))
    if a.check:
        for c in cases:
            if c["id"] in truth:
                print("%2d %-11s %s" % (c["id"], c["cat"], [f[1] for f in truth[c["id"]]]))
        print("\n%d cases, %d with broken ground truth" % (len(cases), bad))
        return
    rows, t0 = [], time.time()
    for i, c in enumerate(cases, 1):
        if c["id"] not in truth:
            continue
        t = time.time()
        try:
            res = agent.ask(c["q"], model=a.model, verbose=False)
        except Exception as e:
            res = {"answer": "CRASH: %s" % e, "trace": [], "grounding": None}
        s = score(c, res, truth[c["id"]])
        rows.append({"id": c["id"], "category": c["cat"], "question": c["q"], **s,
                     "tools_called": " ".join(x["tool"] for x in res["trace"]),
                     "code_fixes": "; ".join(f for x in res["trace"] for f in x.get("fixes", [])),
                     "self_check_used": res.get("self_check_used", False), "secs": round(time.time() - t, 1),
                     "note": c["note"], "answer": res["answer"][:600].replace("\n", " ")})
        print("[%2d/%d] %-4s case %2d %-11s %5.1fs  %s" % (i, len(cases), "PASS" if s["pass"] else "FAIL",
                                                         c["id"], c["cat"], time.time() - t, s["why"][:110]))
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    out = "results/agent_eval%s.csv" % ("" if a.model == "qwen3:8b" else "_" + a.model.replace(":", "_"))
    if not a.only:
        df.to_csv(out, index=False)
    print("\n=== %s, %d cases, %.1f min" % (a.model, len(df), (time.time() - t0) / 60))
    print("pass rate: %.0f%% (%d/%d)" % (100 * df["pass"].mean(), df["pass"].sum(), len(df)))
    for k in ["route", "args", "facts", "grounded"]:
        print("  %-9s %.0f%%" % (k, 100 * df[k].mean()))
    print("  self-check used in %d cases" % df["self_check_used"].sum())
    print("  code fixed arguments in %d cases" % (df["code_fixes"].fillna("") != "").sum())
    print("\nby category:")
    print(df.groupby("category")["pass"].agg(["sum", "count"]).to_string())
    if not a.only:
        print("\nsaved", out)


if __name__ == "__main__":
    main()
