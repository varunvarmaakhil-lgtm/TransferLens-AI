"""TransferLens agent: a local model that chooses tools and explains their results.

Usage:
  python agent.py "Find me a cheaper Gyokeres under 25"
  python agent.py            (interactive; type quit to stop)
  python agent.py --model qwen3:4b "..."
"""
import argparse
import json
import re
import sys
import time
import ollama
import tools

SYSTEM = """You are TransferLens, a football recruitment assistant.
Rules:
1. For any fact or number about a player (club, age, stats, value, price, probability), call a tool. Never answer from memory.
2. Never calculate, compare or invent numbers yourself. Quote numbers exactly as the tools return them, with units (EUR millions, %).
3. Comparing two or more named players: call compare_players and state its verdicts_use_these sentences as they are. "Better value" means the largest gap between fair value and market value, not the lower price. Never rank or compare players yourself; use summary_computed_in_code or the verdicts.
4. A position or role without a reference player (e.g. "centre-backs under 26"): call search_players. A reference player ("like Pavlidis"): call similar_players.
5. Asked why (why he might move, why he is valued so): call explain_player and explain its drivers in plain words, keeping their order and direction. Do not invent other reasons.
5b. Asked how likely a player is to move: the number must come from transfer_probability or a move_probability_pct field. Never state a probability you did not receive.
5c. Club planning ("what does X need", "who should X sign with Y million"): call squad_needs and/or plan_signings. Present the optimiser's signings, totals and constraints as returned; never choose or swap players yourself. If no budget is given for plan_signings, ask for one.
5d. Asked what a player is worth: give fair_value_eur_m first, then the 80% range and market value (use summary_use_this).
5e. Pass player names exactly as the user wrote them; never add or change a first name.
5f. plan_signings: max_signings = the number of players the user asks for ("a striker and a winger" = 2, "two signings" = 2).
6. Do not add filters the user did not ask for. If the user gives no count, use k=5. If the user speaks for a club, pass it as exclude_club.
7. If a tool returns an error such as player_not_found, tell the user and offer the suggested names.
8. Values are Transfermarkt market values, not fees. The fair range is an 80% prediction interval, not an accuracy. Probabilities are model estimates (ROC-AUC about 0.75).
9. Keep answers short. If the question is not about football players, say what you can help with instead."""

MAX_ROUNDS = 5


def numbers(text):
    text = re.sub(r"(?m)^\s*\d+[.)]\s", " ", text)
    text = re.sub(r"(?i)per[\s-]?90|p90|/90", " ", text)
    return [float(x.replace(",", "")) for x in re.findall(r"(?<![\w.])\d[\d,]*\.?\d*", text)]


def check(answer, sources):
    src = set(numbers(" ".join(sources)))
    got = [n for n in numbers(answer) if not (1900 <= n <= 2100 and float(n).is_integer())]

    def near(n):
        return any(abs(c - s) <= max(0.01 * abs(s), 0.051) for s in src for c in (n, n / 100))
    bad = [n for n in got if not near(n)]
    return len(got) - len(bad), len(got), bad


WORDNUM = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "a pair of": "2"}
POS_NOUNS = (r"(goalkeeper|keeper|centre[- ]?back|center[- ]?back|defender|full[- ]?back|left[- ]?back|"
             r"right[- ]?back|midfielder|winger|striker|forward|playmaker|number 10)")
GROUNDED = ("budget_eur_m", "max_age", "max_value_eur_m", "k", "max_signings")


def _digits(q):
    q = q.lower()
    for w, d in WORDNUM.items():
        q = re.sub(r"\b%s\b" % w, d, q)
    return re.sub(r"(?<=[^\d\s.,])(?=\d)", " ", q)


def q_numbers(q):
    ns = numbers(_digits(q))
    ns += [n / 1e6 for n in ns if n >= 1e6]
    if "billion" in q.lower():
        ns += [n * 1000 for n in ns]
    return ns


def requested_count(q):
    qd = _digits(q)
    m = re.search(r"\b(\d)\s+(?:new\s+)?(?:signings|players|additions|transfers)\b", qd)
    if m:
        return int(m.group(1))
    hits = re.findall(r"\b(?:(\d)\s+)?(?:new\s+)?" + POS_NOUNS + r"s?\b", qd)
    total = sum(int(n) if n else 1 for n, _ in hits)
    return total if hits and 1 <= total <= 5 else None


def question_position(q):
    qn = re.sub(r"[^a-z0-9 ]", " ", tools._norm(q).replace("-", " "))
    words, found = qn.split(), set()
    for size in (3, 2, 1):
        for i in range(len(words) - size + 1):
            g = " ".join(words[i:i + size]).rstrip("s")
            if g.endswith("midfielder"):
                g = g[:-2]
            if g == "9" or (len(g) <= 3 and g.upper() not in {w.rstrip("s") for w in re.findall(r"[A-Za-z]+", q)}):
                continue
            if g in tools.ALIASES:
                found.add(tools.ALIASES[g])
            for s in tools.SUB_POSITIONS:
                if tools._norm(s).replace("-", " ") == g:
                    found.add(s)
    return found.pop() if len(found) == 1 else None


def speaking_for(q):
    m = re.search(r"(?i:\b(?:we are|we're|i am|i'm|i work for|i run|as)\s+(?:the\s+)?)"
                  r"([A-Z][\w.&-]*(?:\s+[A-Z][\w.&-]*){0,3})", q)
    name = re.split(r"[.,!?;:]", m.group(1))[0].strip() if m else ""
    if len(name) < 3:
        return None
    try:
        import optimiser
        club, err = optimiser.resolve_club(name)
    except Exception:
        return None
    return None if err else club


def ground_args(name, args, question):
    """Check the model's arguments against the user's own words. Returns (args, notes, error)."""
    args, notes = dict(args), []
    qn = q_numbers(question)
    for k in GROUNDED:
        if k not in args or args[k] in (None, ""):
            continue
        try:
            v = float(args[k])
        except (TypeError, ValueError):
            continue
        if k == "k" and v == 5:
            continue
        tol = 1.0 if k == "max_age" else 0.01 * max(abs(v), 1)
        if any(abs(v - n) <= tol for n in qn):
            continue
        if k == "budget_eur_m":
            return args, notes, {"error": "budget_not_given_by_user", "detail":
                                 "The user did not state a budget. Do not plan; ask the user for their budget."}
        notes.append("dropped %s=%s (not in the question)" % (k, args.pop(k)))
    if name == "plan_signings" and "max_signings" not in args:
        n = requested_count(question)
        if n:
            args["max_signings"] = n
            notes.append("max_signings=%d from the question" % n)
    if name == "search_players":
        if not args.get("sub_position"):
            p = question_position(question)
            if p:
                args["sub_position"] = p
                notes.append("sub_position=%s from the question" % p)
        if not args.get("exclude_club"):
            c = speaking_for(question)
            if c:
                args["exclude_club"] = c
                notes.append("exclude_club=%s (user speaks for this club)" % c)
    return args, notes, None


def ask(question, model="qwen3:8b", verbose=True, self_check=True):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    trace, outputs, t0, rechecked = [], [], time.time(), False
    for _ in range(MAX_ROUNDS):
        r = ollama.chat(model=model, messages=msgs, tools=tools.SCHEMAS, think=False,
                        options={"temperature": 0})
        msg = r.message
        if not msg.tool_calls:
            answer = (msg.content or "").strip()
            ok, total, bad = check(answer, outputs + [question])
            if bad and self_check and not rechecked:
                rechecked = True
                if verbose:
                    print("  [self-check: untraced numbers %s, asking the model to fix]" % bad)
                msgs.append({"role": "assistant", "content": answer})
                msgs.append({"role": "user", "content":
                             "Check failed: these numbers in your answer do not appear in any tool result: %s. "
                             "Call the tool that provides them, or remove them. Reply with only the corrected answer; "
                             "do not mention this check." % bad})
                continue
            if verbose:
                print("\n" + answer)
                gs = "no numbers" if total == 0 else "%d/%d numbers traced to tool outputs" % (ok, total)
                print("\n[%d tool call(s): %s | grounding: %s%s | %.1fs]"
                      % (len(trace), ", ".join(t["tool"] for t in trace) or "none", gs,
                         " | self-check used" if rechecked else "", time.time() - t0))
            return {"answer": answer, "trace": trace, "grounding": (ok, total), "self_check_used": rechecked}
        msgs.append(msg)
        for tc in msg.tool_calls:
            name, raw = tc.function.name, dict(tc.function.arguments or {})
            args, notes, gerr = ground_args(name, raw, question)
            result = gerr if gerr else tools.call(name, args)
            if notes and isinstance(result, dict):
                result = dict(result, argument_fixes_by_code=notes)
            text = json.dumps(result, ensure_ascii=False)
            trace.append({"tool": name, "args": args, "raw_args": raw, "fixes": notes, "error": result.get("error")})
            outputs.append(text)
            if verbose:
                print("  -> %s(%s)%s%s" % (name, ", ".join("%s=%r" % kv for kv in args.items()),
                                           "  [error: %s]" % result["error"] if "error" in result else "",
                                           "  [code fixed: %s]" % "; ".join(notes) if notes else ""))
            msgs.append({"role": "tool", "content": text, "tool_name": name})
    return {"answer": "Stopped after %d rounds without a final answer." % MAX_ROUNDS,
            "trace": trace, "grounding": None, "self_check_used": rechecked}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--model", default="qwen3:8b")
    a = ap.parse_args()
    if a.question:
        ask(" ".join(a.question), a.model)
        sys.exit()
    print("TransferLens agent (%s). Ask about players; quit to stop." % a.model)
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            ask(q, a.model)
