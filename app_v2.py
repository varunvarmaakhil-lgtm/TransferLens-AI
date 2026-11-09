"""TransferLens AI: Streamlit front end.

Run:  ../.venv/bin/streamlit run app_v2.py
"""
import os
import pandas as pd
import streamlit as st
import tools
import optimiser

st.set_page_config(page_title="TransferLens AI", page_icon=":soccer:", layout="wide")
R = "results/"
CC = st.column_config
COLS = {
    "name": CC.TextColumn("Player"), "club": CC.TextColumn("Club"), "from_club": CC.TextColumn("From"),
    "sub_position": CC.TextColumn("Position"), "position": CC.TextColumn("Position"),
    "age": CC.NumberColumn("Age", format="%d"),
    "minutes_share_pct": CC.ProgressColumn("Minutes share", min_value=0, max_value=100, format="%d%%"),
    "goals_plus_assists_per90": CC.NumberColumn("G+A per 90", format="%.2f"),
    "market_value_eur_m": CC.NumberColumn("Market value", format="EUR %.1fM"),
    "fair_mid_eur_m": CC.NumberColumn("Fair value", format="EUR %.1fM"),
    "undervalued": CC.CheckboxColumn("Undervalued"),
    "move_probability_pct": CC.ProgressColumn("Move probability", min_value=0, max_value=100, format="%d%%"),
    "target_depth": CC.NumberColumn("Target"), "reliable_players": CC.NumberColumn("Reliable"),
    "need": CC.NumberColumn("Need"), "likely_to_leave": CC.TextColumn("Likely to leave"),
    "ageing": CC.TextColumn("Ageing (32+)"),
    "season": CC.TextColumn("Season"), "league": CC.TextColumn("League"), "apps": CC.NumberColumn("Apps"),
    "minutes": CC.NumberColumn("Minutes"), "goals": CC.NumberColumn("Goals"), "assists": CC.NumberColumn("Assists"),
}


def table(rows, order):
    df = pd.DataFrame(rows)
    order = [c for c in order if c in df.columns]
    st.dataframe(df, column_order=order, column_config={k: v for k, v in COLS.items() if k in order},
                 hide_index=True, width="stretch")


def fmt_m(x):
    return "-" if x is None else "EUR %.1fM" % x


def read(path):
    try:
        return pd.read_csv(path) if os.path.exists(path) else None
    except Exception:
        return None


def show_career(car):
    if "error" in car or not car["seasons"]:
        return
    st.markdown("**Career**")
    tot = car["totals_covered_seasons"]
    c = st.columns(5)
    c[0].metric("Seasons", tot["seasons"])
    c[1].metric("Appearances", tot["apps"])
    c[2].metric("Goals", tot["goals"])
    c[3].metric("Assists", tot["assists"])
    c[4].metric("Peak market value", fmt_m(tot["peak_market_value_eur_m"]))
    mv = pd.DataFrame(car["seasons"]).set_index("season")["market_value_eur_m"].dropna()
    if len(mv) > 1:
        st.line_chart(mv.rename("Market value (EUR M)"), height=220)
    table(list(reversed(car["seasons"])), ["season", "club", "league", "age", "apps", "minutes", "goals",
                                           "assists", "goals_plus_assists_per90", "market_value_eur_m"])
    st.caption("Clubs: %s. %s" % (" > ".join(tot["clubs"]) or "-", car["note"]))


# ------------------------------------------------------------------ cached work (runs once per input)
@st.cache_data(ttl=30, show_spinner=False)
def llm_status():
    try:
        import ollama
        ollama.list()
        return True, ""
    except Exception as e:
        return False, str(e)[:80]


@st.cache_data(show_spinner=False)
def player_options():
    t = tools._table()
    t = t[t["minutes_share"] >= 0.1].sort_values("mv_end", ascending=False)
    covered = t["name"].drop_duplicates().tolist()
    seen = set(covered)
    return covered, [n for n in tools.outside_players() if n not in seen]


@st.cache_data(show_spinner=False)
def club_options():
    t = tools._table()
    return t.groupby("club_name")["minutes_all"].sum().sort_values(ascending=False).index.tolist()


@st.cache_data(show_spinner=False)
def player_bundle(name):
    p = tools.find_player(name)
    if "error" in p:
        car = tools.career(name)
        return {"error": p} if "error" in car else {"outside": True, "car": car}
    n = p["name"]
    return {"p": p, "fp": tools.fair_price(n), "tp": tools.transfer_probability(n),
            "ex": tools.explain_player(n), "sim": tools.similar_players(n, k=5), "car": tools.career(n)}


@st.cache_data(show_spinner=False)
def club_plan(club, budget, n_sign, positions, max_age, objective):
    needs = optimiser.squad_needs(club)
    if "error" in needs:
        return needs, None
    return needs, optimiser.plan_options(club, budget, n_sign, positions=list(positions) or None,
                                         max_age=max_age or None, objective=objective)


@st.cache_data(show_spinner=False)
def shortlist(sub, max_age, max_val, undervalued, sort_by, k, excl):
    return tools.search_players(sub_position=sub, max_age=max_age, max_value_eur_m=max_val or None,
                                undervalued_only=undervalued, sort_by=sort_by, k=k, exclude_club=excl or None)


# ------------------------------------------------------------------ sidebar
ok, err = llm_status()
with st.sidebar:
    st.title("TransferLens AI")
    st.caption("Football recruitment with evaluated models. Numbers come from models, the optimiser or the data; "
               "the language model only picks tools and explains results.")
    model = st.selectbox("Chat model (local, Ollama)", ["qwen3:8b", "qwen3:4b"])
    st.markdown(("Model: :green[running]" if ok else "Model: :red[not reachable]. Open the Ollama app, then reload."))
    st.divider()
    st.caption("Data: Nodeball Football Datalake (Transfermarkt-based), season 2025/26. "
               "Market values are Transfermarkt estimates, not fees.")

tab_chat, tab_plan, tab_player, tab_list, tab_quality = st.tabs(
    ["Scout chat", "Signing planner", "Player profile", "Shortlist", "Model quality"])

# ------------------------------------------------------------------ chat
EXAMPLES = ["We are Arsenal. Find a defensive midfielder aged 26 or under",
            "Barcelona have 80 million euros. Who should they sign?",
            "Why might Vangelis Pavlidis move this season?",
            "Gyokeres or Pavlidis, who is better value?"]
with tab_chat:
    hist = st.session_state.setdefault("history", [])
    if not hist:
        st.caption("Ask one complete question at a time. Examples:")
        cols = st.columns(len(EXAMPLES))
        for c, ex in zip(cols, EXAMPLES):
            if c.button(ex, width="stretch", disabled=not ok):
                st.session_state["pending"] = ex
    for turn in hist:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("trace"):
                with st.expander("How this answer was made"):
                    st.markdown(turn["trace"])
    q = st.chat_input("Ask about players, values, transfers or a club's window", disabled=not ok)
    q = q or st.session_state.pop("pending", None)
    if q:
        import agent
        hist.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Choosing tools and checking the answer"):
                r = agent.ask(q, model=model, verbose=False)
            st.markdown(r["answer"])
            lines = ["- **%s**(%s)" % (t["tool"], ", ".join("%s=%s" % kv for kv in t["args"].items()))
                     for t in r["trace"]] or ["- no tools needed"]
            fixes = [f for t in r["trace"] for f in t.get("fixes", [])]
            g = r.get("grounding")
            lines.append("- Grounding: " + ("no numbers" if not g or g[1] == 0
                                            else "%d of %d numbers traced to tool outputs" % g))
            if fixes:
                lines.append("- Fixed by code: " + "; ".join(fixes))
            if r.get("self_check_used"):
                lines.append("- Self-check used")
            trace = "\n".join(lines)
            with st.expander("How this answer was made"):
                st.markdown(trace)
        hist.append({"role": "assistant", "content": r["answer"], "trace": trace})
        st.rerun()
    if hist and st.button("Clear chat"):
        st.session_state["history"] = []
        st.rerun()

# ------------------------------------------------------------------ signing planner
with tab_plan:
    clubs = club_options()
    with st.form("plan"):
        c = st.columns([2, 1, 1, 1])
        club = c[0].selectbox("Club", clubs, index=clubs.index("FC Barcelona") if "FC Barcelona" in clubs else 0)
        budget = c[1].number_input("Budget (EUR M)", 5, 500, 80, step=5)
        n_sign = c[2].number_input("Max signings", 1, 5, 3)
        age_cap = c[3].number_input("Max age (0 = any)", 0, 40, 0)
        c = st.columns([3, 1])
        positions = c[0].multiselect("Positions (empty = the club's needs)", tools.SUB_POSITIONS)
        objective = c[1].selectbox("Objective", ["quality", "bargains"],
                                   help="quality = most total fair value; bargains = biggest fair minus market gap")
        go = st.form_submit_button("Plan the window", type="primary")
    if go:
        st.session_state["plan_args"] = (club, float(budget), int(n_sign), tuple(positions), int(age_cap), objective)
    if "plan_args" in st.session_state:
        with st.spinner("Solving"):
            needs, p = club_plan(*st.session_state["plan_args"])
        if "error" in needs:
            st.error("Club not found.")
        else:
            if p is None or "error" in p:
                st.warning((p or {}).get("detail", "No plan found."))
            else:
                st.subheader("Recommended signings")
                st.caption(p["note"])
                order = ["name", "position", "from_club", "age", "market_value_eur_m", "fair_mid_eur_m",
                         "move_probability_pct", "undervalued"]
                for t_, plan in zip(st.tabs(["Plan %s" % "ABC"[i] for i in range(len(p["plans"]))]), p["plans"]):
                    with t_:
                        c = st.columns(4)
                        c[0].metric("Spent", fmt_m(plan["total_cost_eur_m"]),
                                    "%s left" % fmt_m(plan["budget_left_eur_m"]), delta_color="off")
                        c[1].metric("Total fair value", fmt_m(plan["total_fair_value_eur_m"]))
                        c[2].metric("Optimiser score", plan["objective_value"])
                        if "greedy_baseline_objective_value" in plan:
                            gb = plan["greedy_baseline_objective_value"]
                            c[3].metric("Greedy score (same rules)", "no valid plan" if gb is None else gb)
                        table(plan["signings"], order)
                        st.caption("%s. %s. %s." % (plan["objective_meaning"].capitalize(),
                                                    "; ".join(plan["constraints"]).capitalize(), plan["solver"]))
                if p["alternatives"]:
                    with st.expander("More options for these positions (not in any plan)"):
                        table(p["alternatives"], order)
            st.subheader("Squad needs, %s" % needs["club"])
            nd = pd.DataFrame(needs["positions"])
            for col in ("likely_to_leave", "ageing"):
                nd[col] = nd[col].map(lambda v: ", ".join(v) if v else "")
            table(nd, ["position", "target_depth", "reliable_players", "need", "likely_to_leave", "ageing"])
            st.caption("Reliable: %s." % needs["rule"].split("= ", 1)[-1])

# ------------------------------------------------------------------ player profile
with tab_player:
    covered_names, outside_names = player_options()
    outside_set = set(outside_names)
    names = covered_names + outside_names
    default = names.index("Vangelis Pavlidis") if "Vangelis Pavlidis" in names else 0
    name = st.selectbox("Player (type to search)", names, index=default,
                        format_func=lambda n: n + "  (outside covered leagues)" if n in outside_set else n)
    b = player_bundle(name)
    if "error" in b:
        st.error("Player not found.")
    elif b.get("outside"):
        st.subheader(name)
        last = [s for s in b["car"]["seasons"] if s.get("covered")]
        st.warning("%s is not playing in the leagues our data covers this season (for example after a move "
                   "to MLS, Saudi Arabia or Brazil, or retirement), so there is no fair value, move probability or "
                   "similar-player search for him. %s"
                   % (name, ("Last covered season: %s at %s." % (last[-1]["season"], last[-1]["club"])) if last else ""))
        show_career(b["car"])
    else:
        p, fp, tp, ex, sim = b["p"], b["fp"], b["tp"], b["ex"], b["sim"]
        st.subheader(p["name"])
        st.caption("%s | %s | age %s | %d goals, %d assists | %.0f%% of club minutes"
                   % (p["club"], p["sub_position"], p["age"], p["goals"], p["assists"],
                      100 * (p["minutes_share"] or 0)))
        c = st.columns(4)
        c[0].metric("Market value", fmt_m(p["market_value_eur_m"]))
        c[1].metric("Fair value", fmt_m(fp["fair_value_eur_m"]),
                    "range %s-%sM" % (fp["fair_range_eur_m"]["low"], fp["fair_range_eur_m"]["high"]),
                    delta_color="off")
        c[2].metric("Move probability", "%s%%" % tp["probability_changes_club_pct"],
                    "average %s%%" % tp["average_probability_all_players_pct"], delta_color="off")
        c[3].metric("G+A per 90", p["goals_plus_assists_per90"])
        verdict = fp["summary_use_this"].split("; ")[-1].rstrip(".")
        (st.success if fp["undervalued"] else st.info)(verdict.capitalize())
        if "error" not in ex:
            c = st.columns(2)
            with c[0].container(border=True):
                st.markdown("**What drives the move probability**")
                st.markdown("\n".join("- " + s for s in ex["why_move_probability"]))
            with c[1].container(border=True):
                st.markdown("**What drives the fair value**")
                st.markdown("\n".join("- " + s for s in ex["why_fair_value"]))
            st.caption("SHAP contributions, largest first.")
        show_career(b["car"])
        st.markdown("**Similar players**")
        if "error" in sim:
            st.caption(sim.get("detail", sim.get("hint", sim["error"])))
        else:
            rows = [dict(r, minutes_share_pct=round(100 * (r.get("minutes_share") or 0))) for r in sim["players"]]
            table(rows, ["name", "club", "age", "goals_plus_assists_per90", "minutes_share_pct",
                         "market_value_eur_m", "fair_mid_eur_m", "undervalued"])

# ------------------------------------------------------------------ shortlist
with tab_list:
    with st.form("shortlist"):
        c = st.columns(4)
        sub = c[0].selectbox("Position", ["Any"] + tools.SUB_POSITIONS, index=2)
        max_age = c[1].number_input("Max age", 16, 40, 26)
        max_val = c[2].number_input("Max market value (EUR M)", 0, 300, 40, step=5)
        k = c[3].number_input("How many", 5, 15, 10)
        c = st.columns([1, 1, 2])
        sort_by = c[0].selectbox("Sort by", tools.SORTS)
        excl = c[1].selectbox("Exclude club (yours)", ["None"] + club_options())
        undervalued = c[2].checkbox("Undervalued only", True)
        st.form_submit_button("Search", type="primary")
    res = shortlist(None if sub == "Any" else sub, int(max_age), float(max_val), undervalued, sort_by, int(k),
                    None if excl == "None" else excl)
    table(res["players"], ["name", "club", "sub_position", "age", "minutes_share_pct", "goals_plus_assists_per90",
                           "market_value_eur_m", "fair_mid_eur_m", "undervalued", "move_probability_pct"])
    st.caption("Undervalued = market value below the model's 80% fair-value range. "
               "At least 40% of club minutes and EUR 1M market value.")

# ------------------------------------------------------------------ model quality
with tab_quality:
    oe, base, fin = read(R + "optimiser_eval.csv"), read(R + "agent_eval_baseline.csv"), read(R + "agent_eval.csv")
    te, fe = read(R + "transfer_now_eval.csv"), read(R + "fair_price_eval.csv")
    auc = "%.3f" % te["roc_auc"].mean() if te is not None else "-"
    cov = "%.1f%%" % (100 * fe["coverage_80"].mean()) if fe is not None else "-"
    c = st.columns(4)
    c[0].metric("Transfer model ROC-AUC", auc, "was 0.93 with leakage", delta_color="off",
                help="Mean over seasons 2022-2024, each unseen in training")
    c[1].metric("Fair-value range coverage", cov, "target 80%", delta_color="off")
    if oe is not None:
        q = oe.dropna(subset=["greedy"]).query("objective == 'quality'")
        c[2].metric("Optimiser beats greedy", "%.0f%%" % (100 * (q.optimiser > q.greedy + 0.05).mean()),
                    "never worse" if (q.optimiser >= q.greedy - 1e-6).all() else None,
                    delta_color="off")
    if base is not None and fin is not None:
        c[3].metric("Agent questions passed", "%.0f%%" % (100 * fin["pass"].mean()),
                    "%+.0f pts vs model alone" % (100 * (fin["pass"].mean() - base["pass"].mean())))

    with st.expander("Transfer model: the leakage audit"):
        st.write("The original version scored 0.93 on a random split. Two leaks inflated it: 84% of test players "
                 "were also in training, and the label drifted with data coverage. Each fix below is tested on "
                 "seasons the model never saw.")
        st.dataframe(pd.DataFrame({"Version": ["Original: random split, transfer-record label",
                                               "Season-by-season test, label rebuilt from appearances",
                                               "+ 17 time-safe club and playing-time features",
                                               "Shared player-season table",
                                               "Live model: non-club rows removed (coverage fix)"],
                                   "ROC-AUC": ["0.93 (invalid)", "0.642", "0.730", "0.748", auc]}),
                     hide_index=True, width="stretch")
        ev = read(R + "rolling_eval_v2.csv")
        if ev is not None and {"test_season", "cb_roc_auc", "lr_roc_auc"} <= set(ev.columns):
            st.line_chart(ev.set_index("test_season")[["cb_roc_auc", "lr_roc_auc"]])

    with st.expander("Fair-price model"):
        if fe is not None:
            st.dataframe(fe.round(3), hide_index=True, width="stretch")
        st.write("CatBoost quantile model with a conformal 80%% range: real values fell inside it %s of the "
                 "time on unseen seasons." % cov)

    if oe is not None:
        with st.expander("Signing optimiser vs greedy"):
            ok_ = oe.dropna(subset=["greedy"])
            st.dataframe(pd.DataFrame([{
                "Objective": obj, "Scenarios": len(s),
                "Never worse": bool((s.optimiser >= s.greedy - 1e-6).all()),
                "Strictly better": "%.0f%%" % (100 * (s.optimiser > s.greedy + 0.05).mean()),
                "Median gain": "%.1f%%" % s.gain_pct.median()} for obj, s in ok_.groupby("objective")]),
                hide_index=True, width="stretch")
            st.caption("40 clubs x budgets of 20/50/100M x 2 objectives, same rules for both. Greedy found no valid "
                       "plan in %d scenarios. Bargain gains are large percentages of small numbers."
                       % int(oe["greedy"].isna().sum()))

    if base is not None and fin is not None:
        with st.expander("Language-model layer: 52 known-answer questions"):
            st.dataframe(pd.DataFrame([{"Check": k_.capitalize(), "Model alone": "%.0f%%" % (100 * base[k_].mean()),
                                        "With argument checks in code": "%.0f%%" % (100 * fin[k_].mean())}
                                       for k_ in ["pass", "route", "args", "facts", "grounded"]]),
                         hide_index=True, width="stretch")
            st.caption("qwen3:8b, local, temperature 0. Route = right tool; args = right filters; facts = real "
                       "numbers and names in the answer; grounded = every number traceable to a tool output.")
            bad = fin[~fin["pass"]][["id", "question", "why"]]
            if len(bad):
                st.markdown("Failures in the final run")
                st.dataframe(bad, hide_index=True, width="stretch")
