"""Reconstruct customer->brand threads for one brand.
usage: uv run scripts/build_threads.py --brand AmazonHelp
"""
import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def clean(text, brand):
    text = re.sub(rf"@{brand}\b", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", required=True)
    args = ap.parse_args()
    brand = args.brand

    df = pd.read_csv(ROOT / "data/raw/twcs.csv", dtype={"tweet_id": "int64"})
    df["parent"] = pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce")
    df["ts"] = pd.to_datetime(df["created_at"], format="%a %b %d %H:%M:%S %z %Y")
    ids = set(df["tweet_id"])
    roots = df[df["inbound"] & ~df["parent"].isin(ids)]

    # walk up from every brand reply to its root; keep roots that are inbound
    valid = df[df["parent"].isin(ids)]
    parent_of = dict(zip(valid["tweet_id"], valid["parent"].astype("int64")))
    root_of = {}

    def find_root(t):
        seen = []
        while t in parent_of and t not in root_of:
            seen.append(t)
            t = parent_of[t]
        t = root_of.get(t, t)
        for s in seen:
            root_of[s] = t
        return t

    df["root"] = [find_root(t) for t in df["tweet_id"]]
    root_set = set(roots["tweet_id"])
    brand_msgs = df[(df["author_id"] == brand) & df["root"].isin(root_set)]
    thread_roots = set(brand_msgs["root"])
    threads = df[df["root"].isin(thread_roots)].sort_values("ts")

    threads["author"] = (threads["author_id"] == brand).map({True: "brand", False: "customer"})
    threads["clean"] = [clean(t, brand) for t in threads["text"]]
    first = threads[threads["tweet_id"] == threads["root"]].set_index("root")
    reply = threads[threads["author"] == "brand"].drop_duplicates("root").set_index("root")
    turns = threads.groupby("root", sort=False).apply(
        lambda g: list(zip(g["author"], g["clean"])), include_groups=False)
    out = pd.DataFrame({
        "customer_tweet_id": first.index,
        "customer_text": first["clean"],
        "customer_text_raw": first["text"],
        "brand_reply_text": reply["clean"].reindex(first.index),
        "brand_reply_raw": reply["text"].reindex(first.index),
        "full_thread": turns.reindex(first.index),
        "created_at": first["ts"],
    }).sort_values("created_at").reset_index(drop=True)
    outdir = ROOT / "data/processed"
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(outdir / f"{brand}_threads.parquet", index=False)
    out.head(200).to_csv(outdir / f"{brand}_threads_sample.csv", index=False)
    n_turns = out["full_thread"].map(len)
    print(f"{brand}: {len(out)} threads, turns median={n_turns.median()} max={n_turns.max()}, "
          f"cust len median={out['customer_text'].str.len().median()}, "
          f"reply len median={out['brand_reply_text'].str.len().median()}")


if __name__ == "__main__":
    main()
