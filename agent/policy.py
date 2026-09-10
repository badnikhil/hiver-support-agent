"""Deterministic auto-reply vs escalate decision. Rules only, no LLM."""
import re

MIN_CONFIDENCE = 0.6
MIN_RETRIEVAL = 0.28
ESCALATE_INTENTS = {"login_account", "billing_subscription"}

PII = re.compile(r"__email__|__credit_card__|\b(?:\d[ -]?){13,16}\b|\border\s*(?:#|number|no\.?)\s*\w+", re.I)
ANGER = re.compile(r"\b(refund|lawyer|attorney|sue|suing|chargeback|scam|fraud|hacked|cancel+ing|cancel my|worst|"
                   r"fuck\w*|shit\w*|bullshit|wtf|ridiculous|disgust\w*)\b", re.I)
REPEAT = re.compile(r"still waiting|(second|third|fourth|3rd|2nd|4th) time|no (response|reply|answer)|"
                    r"no one (has )?(responded|replied|answered)|(days|weeks) (ago|now)|already (dm|contacted|emailed)", re.I)
NEEDS_ACCOUNT = re.compile(r"\bDM\b|direct message|send us a|account email|username", re.I)


def decide(text, intent, confidence, exemplars, draft):
    """Return {action, reason, signals}. First matching rule gives the reason; all matches go in signals."""
    top = max((e["score"] for e in exemplars), default=0.0)
    checks = [
        (intent in ESCALATE_INTENTS, "account_intent", f"Intent '{intent}' needs a human with account access."),
        (bool(PII.search(text)), "pii", "Message contains PII or an order/card number."),
        (bool(m := ANGER.search(text)), "anger_legal_churn", f"Message contains an anger/legal/churn signal ('{m.group(0) if m else ''}')."),
        (intent == "other" or confidence < MIN_CONFIDENCE, "low_confidence", f"Classifier is unsure (intent={intent}, confidence={confidence:.2f})."),
        (top < MIN_RETRIEVAL, "no_similar_history", f"Nothing similar in history (top retrieval score {top:.2f})."),
        (bool(REPEAT.search(text)), "repeat_contact", "Customer mentions an earlier unanswered contact."),
        (bool(NEEDS_ACCOUNT.search(draft or "")), "needs_account_access", "Draft asks for DM/account details, so a human has to take over."),
    ]
    hits = [(name, reason) for ok, name, reason in checks if ok]
    if hits:
        return {"action": "escalate", "reason": hits[0][1], "signals": [n for n, _ in hits]}
    return {"action": "auto", "reason": "Confident intent, similar resolved history, self-contained draft.", "signals": []}


if __name__ == "__main__":
    ex = [{"score": 0.5}]
    assert decide("songs skip on android", "playback_issue", 0.9, ex, "Hey! Try reinstalling the app /RC")["action"] == "auto"
    assert decide("charged twice", "billing_subscription", 0.9, ex, "x")["signals"] == ["account_intent"]
    assert "pii" in decide("mail me at __email__", "playback_issue", 0.9, ex, "x")["signals"]
    assert "repeat_contact" in decide("still waiting on you", "playback_issue", 0.9, ex, "x")["signals"]
    assert decide("songs skip", "playback_issue", 0.9, ex, "Hey! DM us your account email")["signals"] == ["needs_account_access"]
    assert decide("songs skip", "playback_issue", 0.9, [], "x")["signals"] == ["no_similar_history"]
    print("ok")
