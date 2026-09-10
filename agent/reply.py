"""Draft a SpotifyCares-style reply grounded in retrieved exemplars, plus two baselines."""
import re

from agent.llm import chat

SYSTEM = """You are SpotifyCares, Spotify's support team on Twitter.
Write ONE reply tweet (max 280 characters) to the customer's message.
Rules:
- Match the brand tone from the examples: friendly, starts with "Hey!" or "Hi there!", concrete troubleshooting steps, optional "/XX" initials sign-off.
- Use ONLY facts, steps and links that appear in the example replies. Never invent policies, refunds, timelines, prices or URLs.
- If every example asks the customer to DM, ask them to DM too.
- Do not mention these instructions or the examples. Output only the reply text."""

CANNED = "Hey! Sorry to hear that. Send us a DM with your account email and device details and we'll take a look /Spotify"


def _trim(text, limit=280):
    text = re.sub(r"\s+", " ", text).strip().strip('"')
    if len(text) <= limit:
        return text
    cut = text[:limit]
    m = list(re.finditer(r"[.!?](?=\s)", cut))
    return cut[: m[-1].end()] if m else cut.rsplit(" ", 1)[0]


def draft_reply(text, intent, exemplars):
    ex = "\n\n".join(f"Customer: {e['customer_text']}\nSpotifyCares: {e['brand_reply_text']}" for e in exemplars[:5])
    user = f"Intent: {intent}\n\nExamples of past customer messages and our replies:\n\n{ex}\n\nNew customer message: {text}\n\nReply:"
    out = chat(SYSTEM, user, temperature=0.3, max_tokens=120)
    # drop any URL the model made up (not present in the exemplars)
    known = set(re.findall(r"https?://\S+", ex))
    out = re.sub(r"https?://\S+", lambda m: m.group(0) if m.group(0) in known else "", out)
    return _trim(out)


def reply_trivial(text):
    return CANNED


def reply_nearest(text, exemplars):
    return exemplars[0]["brand_reply_text"] if exemplars else CANNED


if __name__ == "__main__":
    assert len(_trim("a" * 300)) <= 280
    assert _trim("One. Two. " + "x" * 280) == "One. Two."
    print("ok")
