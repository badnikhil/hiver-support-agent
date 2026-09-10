"""LLM-label corpus_sample (minus golden candidates) -> data/processed/silver_labels.jsonl.
usage: uv run scripts/silver_labels.py   (resumable: cached LLM calls are free)
"""
import collections
import json
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent.intents import classify  # noqa: E402


def main():
    sample = pd.read_json(ROOT / "data/processed/corpus_sample.jsonl", lines=True)
    golden = pd.read_json(ROOT / "data/golden/candidates.jsonl", lines=True)
    sample = sample[~sample["customer_tweet_id"].isin(golden["customer_tweet_id"])]
    t0 = time.time()
    with open(ROOT / "data/processed/silver_labels.jsonl", "w") as f:
        for tid, text in tqdm(zip(sample["customer_tweet_id"], sample["customer_text"]), total=len(sample)):
            f.write(json.dumps({"customer_tweet_id": int(tid), "customer_text": text, **classify(text)},
                               ensure_ascii=False) + "\n")
    labels = pd.read_json(ROOT / "data/processed/silver_labels.jsonl", lines=True)
    print(f"{len(labels)} rows, {(time.time()-t0)/len(labels):.2f}s/row")
    print(labels["intent"].value_counts().to_string())


if __name__ == "__main__":
    main()
