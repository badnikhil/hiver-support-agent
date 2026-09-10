"""Per-brand stats for the top N brands in twcs.csv. Writes agent-docs/brand_stats.md."""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DM_RE = re.compile(r"\b(?:DM|direct message|private message)\b", re.I)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 15


def main():
    df = pd.read_csv(ROOT / "data/raw/twcs.csv", dtype={"tweet_id": "int64"})
    df["parent"] = pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce")
    ids = set(df["tweet_id"])
    df["is_root"] = ~df["parent"].isin(ids)
    by_id = df.set_index("tweet_id")
    brands = df.loc[~df["inbound"], "author_id"].value_counts().head(N).index

    rows, samples = [], {}
    for b in brands:
        replies = df[df["author_id"] == b]
        parents = by_id.reindex(replies["parent"].dropna().astype("int64"))
        cust_parents = parents[parents["inbound"] == True]  # noqa: E712
        mention = df["inbound"] & df["text"].str.contains(f"@{b}\\b", case=False, regex=True)
        rows.append({
            "brand": b,
            "brand_replies": len(replies),
            "inbound_tweets": int(mention.sum()),
            "threads": int((cust_parents["is_root"]).sum()),
            "med_cust_len": int(df.loc[mention, "text"].str.len().median()),
            "dm_pct": round(100 * replies["text"].str.contains(DM_RE).mean(), 1),
            "en_pct": round(100 * df.loc[mention, "text"].map(str.isascii).mean(), 1),
        })
        pairs = replies[replies["parent"].isin(cust_parents.index)].sample(3, random_state=0)
        samples[b] = [(cust_parents.loc[p, "text"], t) for p, t in zip(pairs["parent"], pairs["text"])]

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    out = ROOT / "agent-docs/brand_stats.md"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        f.write("# Brand stats (top %d brands by reply count)\n\n" % N)
        f.write(table.to_markdown(index=False) + "\n\n")
        for b, ps in samples.items():
            f.write(f"## {b}\n\n")
            for c, r in ps:
                f.write(f"- C: {c!r}\n  B: {r!r}\n")
            f.write("\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
