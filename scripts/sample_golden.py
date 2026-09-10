"""Pick 220 golden-set candidates from corpus.parquet, disjoint from corpus_sample.
200 stratified by month + 20 hard cases. usage: uv run scripts/sample_golden.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HARD = r"cancel|refund|lawyer|hacked|scam|sue|lawsuit"


def main():
    df = pd.read_parquet(ROOT / "data/processed/corpus.parquet")
    silver = pd.read_json(ROOT / "data/processed/corpus_sample.jsonl", lines=True)
    pool = df[~df["customer_tweet_id"].isin(silver["customer_tweet_id"])].copy()
    # months before Sep 2017 have <25 threads each; pool them into one stratum
    month = pool["created_at"].dt.strftime("%Y-%m")
    pool["stratum"] = month.where(month >= "2017-09", "pre-2017-09")
    strata = pool["stratum"].unique()
    per = 200 // len(strata)
    parts = [g.sample(min(per, len(g)), random_state=42) for _, g in pool.groupby("stratum")]
    main_ = pd.concat(parts)
    if len(main_) < 200:
        rest = pool.drop(main_.index).sample(200 - len(main_), random_state=42)
        main_ = pd.concat([main_, rest])
    rest = pool.drop(main_.index)
    txt = rest["customer_text"]
    hard = pd.concat([
        rest[txt.str.len() < 30].sample(5, random_state=42),                       # very short
        rest[txt.str.contains(HARD, case=False)].sample(6, random_state=42),       # angry/legal
        rest[txt.str.contains(r"\balso\b|\band also\b|another (?:issue|problem)", case=False)]
            .sample(5, random_state=42),                                           # two issues
        rest[~txt.str.contains(r"\?|help|can't|cant|won't|not|issue|problem", case=False)]
            .sample(4, random_state=42),                                           # non-support chatter
    ])
    out = pd.concat([main_, hard]).drop_duplicates("customer_tweet_id")
    cols = ["customer_tweet_id", "customer_text", "brand_reply_text", "full_thread", "created_at"]
    out[cols].to_json(ROOT / "data/golden/candidates.jsonl", orient="records", lines=True,
                      force_ascii=False, date_format="iso")
    print(len(out), "candidates;", main_["stratum"].value_counts().to_dict())


if __name__ == "__main__":
    main()
