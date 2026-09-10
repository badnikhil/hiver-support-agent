"""End-to-end eval on the golden set.
usage: uv run -m eval.run_eval [--quick] [--n N] [--skip-judge] [--golden path]
Writes data/eval/predictions.jsonl (resume-safe), results.md, results.json.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support
from tabulate import tabulate
from tqdm import tqdm

from agent.intents import INTENTS
from agent.pipeline import run_batch
from agent.reply import reply_nearest, reply_trivial
from eval.baselines import ESCALATION_BASELINES, intent_baselines
from eval.judge import DIMS, judge

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/eval"
SYSTEMS = ["agent", "reply_trivial", "reply_nearest"]


def load(path):
    return [json.loads(l) for l in open(path)] if Path(path).exists() else []


def esc(labels):
    return ["escalate" if x else "auto" for x in labels]


def prf(y_true, y_pred, pos):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[pos], zero_division=0)
    return {"precision": round(p[0], 3), "recall": round(r[0], 3), "f1": round(f[0], 3)}


def intent_block(rows, preds):
    truth = [r["intent"] for r in rows]
    systems = {"agent": [preds[r["customer_tweet_id"]]["intent"] for r in rows]}
    for name, fn in intent_baselines().items():
        systems[name] = [fn(r["customer_text"]) for r in rows]
    res = {}
    for name, yp in systems.items():
        p, r, f, s = precision_recall_fscore_support(truth, yp, labels=INTENTS, zero_division=0)
        res[name] = {"accuracy": round(accuracy_score(truth, yp), 3),
                     "macro_f1": round(f1_score(truth, yp, average="macro", labels=INTENTS, zero_division=0), 3),
                     "per_intent": {i: {"p": round(p[k], 2), "r": round(r[k], 2), "f1": round(f[k], 2), "n": int(s[k])} for k, i in enumerate(INTENTS)},
                     "confusion": confusion_matrix(truth, yp, labels=INTENTS).tolist()}
    return res


def escalation_block(rows, preds):
    truth = esc([r["should_escalate"] for r in rows])
    systems = {"agent": [preds[r["customer_tweet_id"]]["action"] for r in rows]}
    for name, fn in ESCALATION_BASELINES.items():
        systems[name] = [fn(r["customer_text"]) for r in rows]
    res = {}
    for name, yp in systems.items():
        unsafe = sum(t == "escalate" and p == "auto" for t, p in zip(truth, yp))
        res[name] = {"escalate": prf(truth, yp, "escalate"), "auto": prf(truth, yp, "auto"),
                     "unsafe_auto": unsafe, "unsafe_auto_rate": round(unsafe / len(rows), 3),
                     "escalate_share": round(yp.count("escalate") / len(rows), 3)}
    first = Counter(preds[r["customer_tweet_id"]]["signals"][0] for r in rows if preds[r["customer_tweet_id"]]["signals"])
    every = Counter(s for r in rows for s in preds[r["customer_tweet_id"]]["signals"])
    return res, {"first_reason": dict(first.most_common()), "any_signal": dict(every.most_common())}


def judge_block(rows, preds):
    scores = {s: [] for s in SYSTEMS}  # system -> list of {customer_tweet_id, **dims}
    for r in tqdm(rows, desc="judge"):
        p = preds[r["customer_tweet_id"]]
        replies = {"agent": p["draft"], "reply_trivial": reply_trivial(r["customer_text"]),
                   "reply_nearest": reply_nearest(r["customer_text"], p["exemplars"])}
        for s, reply in replies.items():
            scores[s].append({"customer_tweet_id": r["customer_tweet_id"], **judge(r["customer_text"], reply, r["brand_reply_text"], p["exemplars"])})
    summary = {}
    for s, lst in scores.items():
        summary[s] = {d: round(float(np.mean([x[d] for x in lst])), 2) for d in DIMS}
        summary[s]["pct_overall_ge4"] = round(100 * np.mean([x["overall"] >= 4 for x in lst]), 1)
    wins = {}
    for b in SYSTEMS[1:]:
        pairs = [(a["overall"], c["overall"]) for a, c in zip(scores["agent"], scores[b])]
        wins[b] = {"win": sum(a > c for a, c in pairs), "tie": sum(a == c for a, c in pairs), "loss": sum(a < c for a, c in pairs),
                   "win_rate": round(sum(a > c for a, c in pairs) / len(pairs), 3)}
    return {"n": len(rows), "mean": summary, "agent_vs": wins}, scores


def agreement_block(rows):
    b = {r["customer_tweet_id"]: r for r in load(ROOT / "data/golden/golden_annotator_b.jsonl")}
    both = [(r, b[r["customer_tweet_id"]]) for r in rows if r["customer_tweet_id"] in b]
    if not both:
        return None
    ai, bi = [x["intent"] for x, _ in both], [y["intent"] for _, y in both]
    ae, be = [bool(x["should_escalate"]) for x, _ in both], [bool(y["should_escalate"]) for _, y in both]
    return {"n": len(both),
            "intent": {"kappa": round(cohen_kappa_score(ai, bi), 3), "raw": round(np.mean([x == y for x, y in zip(ai, bi)]), 3)},
            "should_escalate": {"kappa": round(cohen_kappa_score(ae, be), 3), "raw": round(np.mean([x == y for x, y in zip(ae, be)]), 3)}}


def human_block(judge_scores):
    human = load(OUT / "human_scores.jsonl")
    if not human or not judge_scores:
        return None
    key = {r["id"]: r for r in load(OUT / "human_key.jsonl")}
    for h in human:  # template rows carry an id, not the system; the key maps it back
        if "system" not in h and h.get("id") in key:
            h.update(customer_tweet_id=key[h["id"]]["customer_tweet_id"], system=key[h["id"]]["system"])
    j = {(x["customer_tweet_id"], s): x for s, lst in judge_scores.items() for x in lst}
    pairs = [(h, j[(h["customer_tweet_id"], h["system"])]) for h in human
             if h.get("overall") is not None and (h.get("customer_tweet_id"), h.get("system")) in j]
    if len(pairs) < 3:
        return {"n": len(pairs), "note": "not enough scored rows overlapping the judged set"}
    hv, jv = [int(h["overall"]) for h, _ in pairs], [int(x["overall"]) for _, x in pairs]
    out = {"n": len(pairs), "overall": {
        "spearman": round(float(spearmanr(hv, jv).statistic), 3),
        "qwk": round(cohen_kappa_score(hv, jv, weights="quadratic"), 3),
        "exact": round(np.mean([a == b for a, b in zip(hv, jv)]), 3),
        "within1": round(np.mean([abs(a - b) <= 1 for a, b in zip(hv, jv)]), 3),
        "acceptable_kappa": round(cohen_kappa_score([a >= 4 for a in hv], [b >= 4 for b in jv]), 3)}}
    for d in DIMS[:-1]:
        dp = [(int(h[d]), int(x[d])) for h, x in pairs if h.get(d) is not None]
        if len(dp) >= 3:
            out[d] = {"spearman": round(float(spearmanr(*zip(*dp)).statistic), 3), "n": len(dp)}
    return out


def to_md(R):
    L = [f"# Eval results (n={R['n']})\n", "## Intent\n"]
    L.append(tabulate([[s, v["accuracy"], v["macro_f1"]] for s, v in R["intent"].items()], ["system", "accuracy", "macro-F1"], tablefmt="github"))
    ag = R["intent"]["agent"]["per_intent"]
    L.append("\n### Agent per intent\n" + tabulate([[i, v["p"], v["r"], v["f1"], v["n"]] for i, v in ag.items()], ["intent", "P", "R", "F1", "n"], tablefmt="github"))
    L.append("\n### Agent confusion (rows = gold, cols = predicted)\n" + tabulate(
        [[i] + row for i, row in zip(INTENTS, R["intent"]["agent"]["confusion"])], ["gold \\ pred"] + [i[:8] for i in INTENTS], tablefmt="github"))
    L.append("\n## Escalation\n" + tabulate(
        [[s, v["escalate"]["precision"], v["escalate"]["recall"], v["escalate"]["f1"], v["auto"]["precision"], v["auto"]["recall"], v["auto"]["f1"],
          v["unsafe_auto"], v["unsafe_auto_rate"], v["escalate_share"]] for s, v in R["escalation"].items()],
        ["system", "esc P", "esc R", "esc F1", "auto P", "auto R", "auto F1", "unsafe auto", "unsafe rate", "esc share"], tablefmt="github"))
    L.append("\nunsafe auto = predicted auto when the human said escalate.\n\n### Agent escalation reasons (first rule hit)\n" + tabulate(
        list(R["reasons"]["first_reason"].items()), ["reason", "n"], tablefmt="github"))
    L.append("\n### All signals fired\n" + tabulate(list(R["reasons"]["any_signal"].items()), ["signal", "n"], tablefmt="github"))
    if R.get("judge"):
        J = R["judge"]
        L.append(f"\n## Reply quality (LLM judge, n={J['n']})\n" + tabulate(
            [[s] + [v[d] for d in DIMS] + [v["pct_overall_ge4"]] for s, v in J["mean"].items()], ["system"] + DIMS + ["% overall>=4"], tablefmt="github"))
        L.append("\n### Paired: agent vs baseline on overall\n" + tabulate(
            [[b, v["win"], v["tie"], v["loss"], v["win_rate"]] for b, v in J["agent_vs"].items()], ["baseline", "win", "tie", "loss", "win rate"], tablefmt="github"))
    if R.get("agreement"):
        A = R["agreement"]
        L.append(f"\n## Inter-annotator agreement (n={A['n']})\n" + tabulate(
            [[k, v["kappa"], v["raw"]] for k, v in A.items() if k != "n"], ["field", "Cohen kappa", "raw agreement"], tablefmt="github"))
    if R.get("judge_vs_human"):
        H = R["judge_vs_human"]
        L.append(f"\n## Judge vs human (n={H['n']})\n" + (H.get("note") or tabulate(
            [[k, v] for k, v in H["overall"].items()], ["overall metric", "value"], tablefmt="github")))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=ROOT / "data/golden/golden.jsonl")
    ap.add_argument("--n", type=int, help="judge only this many rows (seeded subset)")
    ap.add_argument("--quick", action="store_true", help="30 rows for everything")
    ap.add_argument("--skip-judge", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load(a.golden)
    if a.quick:
        rows = random.Random(42).sample(rows, min(30, len(rows)))
    preds = run_batch(rows, out / "predictions.jsonl")
    R = {"n": len(rows), "intent": intent_block(rows, preds)}
    R["escalation"], R["reasons"] = escalation_block(rows, preds)
    R["agreement"] = agreement_block(rows)
    scores = None
    if not a.skip_judge:
        jrows = random.Random(42).sample(rows, min(a.n, len(rows))) if a.n else rows
        R["judge"], scores = judge_block(jrows, preds)
        (out / "judge_scores.json").write_text(json.dumps(scores, indent=1))
    R["judge_vs_human"] = human_block(scores)
    (out / "results.json").write_text(json.dumps(R, indent=1))
    (out / "results.md").write_text(to_md(R))
    print(f"n={R['n']}  intent acc={R['intent']['agent']['accuracy']} macroF1={R['intent']['agent']['macro_f1']}  "
          f"(tfidf_lr acc={R['intent']['tfidf_lr']['accuracy']}, keyword acc={R['intent']['keyword']['accuracy']})")
    e = R["escalation"]["agent"]
    print(f"escalate P/R/F1={e['escalate']['precision']}/{e['escalate']['recall']}/{e['escalate']['f1']}  "
          f"unsafe auto={e['unsafe_auto']} ({e['unsafe_auto_rate']})  escalate share={e['escalate_share']}")
    if R.get("judge"):
        for s, v in R["judge"]["mean"].items():
            print(f"judge {s:14s} overall={v['overall']} grounded={v['grounded']} %>=4={v['pct_overall_ge4']}")
        print("agent win rate:", {b: v["win_rate"] for b, v in R["judge"]["agent_vs"].items()})
    if R.get("agreement"):
        print("IAA:", R["agreement"])
    if R.get("judge_vs_human"):
        print("judge vs human:", R["judge_vs_human"])
    print(f"-> {out/'results.md'}")


if __name__ == "__main__":
    main()
