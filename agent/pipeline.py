"""classify -> retrieve -> draft -> decide.
usage: uv run -m agent.pipeline "my songs keep skipping on android"
       uv run -m agent.pipeline --jsonl in.jsonl --out out.jsonl   (rows: tweet_id+text or corpus columns; resume-safe)
"""
import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from agent.intents import classify
from agent.policy import decide
from agent.reply import draft_reply
from agent.retrieval import retrieve


def handle(text, exclude_id=None):
    t0 = time.time()
    c = classify(text)
    exemplars = retrieve(text, k=5, exclude_id=exclude_id)
    draft = draft_reply(text, c["intent"], exemplars)
    d = decide(text, c["intent"], c["confidence"], exemplars, draft)
    return {"text": text, "intent": c["intent"], "confidence": c["confidence"], "exemplars": exemplars,
            "draft": draft, "action": d["action"], "reason": d["reason"], "signals": d["signals"],
            "seconds": round(time.time() - t0, 2)}


def show(r):
    print(f"TEXT: {r['text']}\nINTENT: {r['intent']} ({r['confidence']:.2f})")
    for e in r["exemplars"][:3]:
        print(f"  [{e['score']:.2f}{' DM' if e['is_redirect'] else ''}] {e['customer_text'][:90]}\n      -> {e['brand_reply_text'][:120]}")
    print(f"DRAFT: {r['draft']}\nACTION: {r['action']} — {r['reason']}\n({r['seconds']}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("--jsonl")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.text:
        return show(handle(a.text))
    out = Path(a.out)
    done = {json.loads(l)["tweet_id"] for l in out.open()} if out.exists() else set()
    rows = [json.loads(l) for l in open(a.jsonl)]
    rows = [{"tweet_id": r.get("tweet_id", r.get("customer_tweet_id")), "text": r.get("text", r.get("customer_text"))} for r in rows]
    with out.open("a") as f:
        for row in tqdm([r for r in rows if r["tweet_id"] not in done]):
            r = handle(row["text"], exclude_id=row["tweet_id"])
            f.write(json.dumps({"tweet_id": row["tweet_id"], **r}) + "\n")
            f.flush()


if __name__ == "__main__":
    main()
