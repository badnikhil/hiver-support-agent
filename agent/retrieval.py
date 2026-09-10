"""TF-IDF retrieval of similar historical customer messages + the brand's actual reply."""
import re
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data/processed/corpus.parquet"
INDEX = ROOT / "data/index/tfidf.joblib"

# ponytail: TF-IDF only. Ceiling: no synonyms/paraphrase ("songs skip" vs "tracks stutter").
# Upgrade path: sentence-transformers (all-MiniLM) embeddings + cosine, same retrieve() signature.

REDIRECT = re.compile(r"\bDM\b|direct message|send us a", re.I)
_idx = None


def is_redirect(reply):
    return bool(reply) and len(reply) < 80 and bool(REDIRECT.search(reply))


def build_index():
    df = pd.read_parquet(CORPUS)
    df = df.dropna(subset=["customer_text", "brand_reply_text"])
    df = df[df["customer_text"].str.len() >= 15].reset_index(drop=True)
    df["brand_reply_text"] = df["brand_reply_text"].str.replace(r"^(@\w+\s*)+", "", regex=True)
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=200_000)
    mat = hstack([word.fit_transform(df["customer_text"]), char.fit_transform(df["customer_text"])]).tocsr()
    meta = df[["customer_tweet_id", "customer_text", "brand_reply_text"]].copy()
    meta["is_redirect"] = meta["brand_reply_text"].map(is_redirect)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"word": word, "char": char, "mat": mat, "meta": meta}, INDEX)
    return len(meta)


def _load():
    global _idx
    if _idx is None:
        _idx = joblib.load(INDEX)
    return _idx


def retrieve(text, k=5, exclude_id=None):
    """Top-k similar historical customer messages. Redirect replies are ranked after substantive ones."""
    ix = _load()
    q = hstack([ix["word"].transform([text]), ix["char"].transform([text])]).tocsr()
    scores = (ix["mat"] @ q.T).toarray().ravel() / 2  # each block is L2-normalised, so /2 keeps cosine in [0,1]
    meta = ix["meta"]
    if exclude_id is not None:
        scores[(meta["customer_tweet_id"] == exclude_id).to_numpy()] = -1
    top = scores.argsort()[::-1][: k * 3]
    hits = [{"customer_text": meta.customer_text.iat[i], "brand_reply_text": meta.brand_reply_text.iat[i],
             "score": float(scores[i]), "is_redirect": bool(meta.is_redirect.iat[i])} for i in top if scores[i] > 0]
    hits.sort(key=lambda h: (h["is_redirect"], -h["score"]))  # substantive first, stable by score
    return hits[:k]


if __name__ == "__main__":
    for h in retrieve("my songs keep skipping on android", 3):
        print(f"{h['score']:.2f} redirect={h['is_redirect']} | {h['customer_text'][:80]} -> {h['brand_reply_text'][:80]}")
