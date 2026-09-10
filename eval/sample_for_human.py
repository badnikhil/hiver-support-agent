"""60 golden rows x {agent draft, reply_nearest} -> 120 blind items for human scoring.
usage: uv run -m eval.sample_for_human   (needs data/eval/predictions.jsonl)
Writes data/eval/human_scores_template.jsonl (blind) and data/eval/human_key.jsonl (id -> system).
Fill the template's score fields and save as data/eval/human_scores.jsonl.
"""
import json
import random
from pathlib import Path

from agent.reply import reply_nearest
from eval.judge import human_template

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/eval"


def main():
    rng = random.Random(42)
    golden = [json.loads(l) for l in open(ROOT / "data/golden/golden.jsonl")]
    preds = {r["tweet_id"]: r for r in map(json.loads, open(OUT / "predictions.jsonl"))}
    rows = rng.sample([g for g in golden if g["customer_tweet_id"] in preds], 60)
    items, key = [], []
    for g in rows:
        p = preds[g["customer_tweet_id"]]
        for system, reply in [("agent", p["draft"]), ("reply_nearest", reply_nearest(g["customer_text"], p["exemplars"]))]:
            i = f"{g['customer_tweet_id']}-{rng.randrange(1000, 9999)}"
            items.append((i, g["customer_text"], reply))
            key.append({"id": i, "customer_tweet_id": g["customer_tweet_id"], "system": system})
    order = list(range(len(items)))
    rng.shuffle(order)
    with open(OUT / "human_scores_template.jsonl", "w") as f:
        for r in human_template([items[i] for i in order]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT / "human_key.jsonl", "w") as f:
        for i in order:
            f.write(json.dumps(key[i]) + "\n")
    print(f"{len(items)} items -> {OUT/'human_scores_template.jsonl'}, key -> {OUT/'human_key.jsonl'}")


if __name__ == "__main__":
    main()
