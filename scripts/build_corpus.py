"""Clean SpotifyCares threads -> corpus.parquet + 3000-thread committed sample.
usage: uv run scripts/build_corpus.py
"""
import html
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SIG = re.compile(r"\s*(?:[/^*~-]\s?[A-Z]{1,3}|\(\d/\d\)|\d/\d)(?=\s|$)")
COLS = ["customer_tweet_id", "customer_text", "brand_reply_text", "full_thread", "created_at"]


def clean_reply(text):
    return re.sub(r"\s+", " ", SIG.sub("", html.unescape(text))).strip()


def english_ish(s):
    return sum(ch.isascii() for ch in s) / max(len(s), 1) >= 0.85


def main():
    df = pd.read_parquet(ROOT / "data/processed/SpotifyCares_threads.parquet")
    n0 = len(df)
    df["customer_text"] = df["customer_text"].map(html.unescape)
    df = df[(df["customer_text"].str.len() >= 15) & df["customer_text"].map(english_ish)]
    df = df[df["brand_reply_text"].notna()].copy()
    df["brand_reply_text"] = df["brand_reply_text"].map(clean_reply)
    df["full_thread"] = df["full_thread"].map(
        lambda t: [(a, clean_reply(x) if a == "brand" else html.unescape(x)) for a, x in t])
    df = df[COLS].reset_index(drop=True)
    df.to_parquet(ROOT / "data/processed/corpus.parquet", index=False)
    sample = df.sample(3000, random_state=42)
    sample.to_json(ROOT / "data/processed/corpus_sample.jsonl", orient="records", lines=True,
                   force_ascii=False, date_format="iso")
    print(f"threads {n0} -> clean {len(df)}, sample {len(sample)}")


if __name__ == "__main__":
    main()
