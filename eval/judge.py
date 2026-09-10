"""LLM-as-judge for a drafted reply, plus the blank template for human scoring."""
import json
import os

from agent.llm import chat

MODEL = os.environ.get("JUDGE_MODEL")  # None -> agent.llm default
DIMS = ["grounded", "helpful", "tone", "safe", "overall"]

SYSTEM = """You are grading a candidate reply from SpotifyCares (Spotify's support account on Twitter).
You get: the customer's tweet, the CANDIDATE reply to grade, the REFERENCE reply a real agent sent, and EXAMPLE replies the brand sent to similar tweets.
Score the CANDIDATE only. Integers 1-5 for each:
- grounded: 5 = every step, claim and link in the candidate appears in the reference or examples; 3 = mostly grounded but adds a small unsupported detail; 1 = invents policy, steps or links.
- helpful: 5 = addresses the customer's actual issue with an actionable next step; 3 = generic but relevant; 1 = ignores the issue or answers a different one.
- tone: 5 = sounds like SpotifyCares on Twitter: warm, concise, under 280 characters, no corporate filler; 3 = fine but stiff or wordy; 1 = rude, robotic or way too long.
- safe: 5 = no promises of refunds, timelines or features, never asks for passwords or card numbers publicly; 3 = a vague promise ("we'll fix it"); 1 = promises a refund/date/feature or asks for a password/card number.
- overall: 5 = could be sent as-is; 3 = needs a light edit; 1 = must not be sent.
Reply with JSON only: {"grounded": n, "helpful": n, "tone": n, "safe": n, "overall": n}"""


def judge(message, reply, reference, exemplars, model=MODEL):
    ex = "\n".join(f"- {e['brand_reply_text']}" for e in exemplars[:5]) or "- (none)"
    user = f"CUSTOMER: {message}\n\nCANDIDATE: {reply}\n\nREFERENCE: {reference}\n\nEXAMPLES:\n{ex}"
    raw = chat(SYSTEM, user, json_mode=True, temperature=0, max_tokens=80, model=model)
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        out = {}
    scores = {}
    for d in DIMS:
        try:
            scores[d] = min(5, max(1, int(out.get(d, 1))))
        except (TypeError, ValueError):
            scores[d] = 1
    return scores


def human_template(items):
    """items: (id, message, reply) -> rows with blank score fields for a human to fill."""
    return [{"id": i, "message": m, "reply": r, **{d: None for d in DIMS}, "comment": ""} for i, m, r in items]


if __name__ == "__main__":
    s = judge("my songs keep skipping on android", "Hey! Try reinstalling the app and let us know how it goes /SC",
              "Hey! Could you try a clean reinstall? Steps: https://t.co/x", [{"brand_reply_text": "Hey! Try a clean reinstall https://t.co/x"}])
    assert set(s) == set(DIMS) and all(1 <= v <= 5 for v in s.values())
    print(s)
