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
            name, args = tc.function.name, dict(tc.function.arguments or {})
            result = tools.call(name, args)
            text = json.dumps(result, ensure_ascii=False)
            trace.append({"tool": name, "args": args, "error": result.get("error")})
            outputs.append(text)
            if verbose:
                print("  -> %s(%s)%s" % (name, ", ".join("%s=%r" % kv for kv in args.items()),
                                         "  [error: %s]" % result["error"] if "error" in result else ""))
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
