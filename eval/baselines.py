"""Cheap intent / escalation baselines to compare the agent against."""
import json
import re
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

SILVER = Path(__file__).resolve().parent.parent / "data/processed/silver_labels.jsonl"

# first match wins, in this order; nothing matches -> other
KEYWORDS = {
    "login_account": r"log ?in|password|hacked|facebook|sign ?in|locked out",
    "billing_subscription": r"charg|refund|premium|subscri|cancel|trial|student|family|paid|payment",
    "downloads_library": r"download|offline|library|playlists? (gone|disappear|missing|deleted)|disappear",
    "device_connect": r"chromecast|sonos|alexa|bluetooth|\bcar\b|\btv\b|xbox|playstation|connect|speaker",
    "content_missing": r"not available|unavailable|grey|region|country|where('s| is| are)|album|release",
    "playback_issue": r"crash|won'?t play|not playing|skip|paus|stutter|freez|glitch|bug|error|loading",
    "feature_feedback": r"feature|should|please add|wish|suggest|love|thank|shuffle|discover weekly|\bui\b",
}
KEYWORDS = {k: re.compile(v, re.I) for k, v in KEYWORDS.items()}
ESCALATE_KW = re.compile(r"refund|cancel|charged|login|password|hacked|account", re.I)


def load_silver():
    return [json.loads(l) for l in SILVER.open()]


def keyword(text):
    return next((k for k, rx in KEYWORDS.items() if rx.search(text)), "other")


def intent_baselines(silver=None):
    """name -> fn(text) -> intent. tfidf_lr is fit here (seed 42)."""
    silver = silver or load_silver()
    top = Counter(r["intent"] for r in silver).most_common(1)[0][0]
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                        LogisticRegression(max_iter=1000, C=5, random_state=42))
    clf.fit([r["customer_text"] for r in silver], [r["intent"] for r in silver])
    return {"majority": lambda t: top, "keyword": keyword, "tfidf_lr": lambda t: str(clf.predict([t])[0])}


ESCALATION_BASELINES = {
    "always_escalate": lambda t: "escalate",
    "never_escalate": lambda t: "auto",
    "keyword_escalate": lambda t: "escalate" if ESCALATE_KW.search(t) else "auto",
}


if __name__ == "__main__":
    assert keyword("I was charged twice") == "billing_subscription"
    assert keyword("my downloads vanished") == "downloads_library"
    assert keyword("lol") == "other"
    assert ESCALATION_BASELINES["keyword_escalate"]("someone hacked me") == "escalate"
    b = intent_baselines()
    print({k: f(" my songs keep skipping") for k, f in b.items()})
