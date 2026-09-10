"""Intent taxonomy + LLM classifier for SpotifyCares customer tweets."""
import json

from agent.llm import chat

# intent -> (one-line definition, few-shot examples from real data)
TAXONOMY = {
    "playback_issue": (
        "app crashes, errors, songs won't play/skip/pause/stutter, slow loading, high CPU, ads misbehaving",
        ["So my app won't play music.. fix it jesus",
         "when I try to share a song from mobile, the app crashes",
         "why does my music pause every time I get a notification"]),
    "login_account": (
        "can't log in, hacked, password/email reset, Facebook link, duplicate/delete accounts, account recovery",
        ["I can't log into my Spotify, the reset password email won't send",
         "I just received an email saying my email address has been changed and now I cannot log in",
         "How do I remove my Facebook account from my Spotify account?"]),
    "billing_subscription": (
        "charged wrongly, refund, cancel, paid but still Free, trial length, student/family/duo plans, gift cards, payment methods",
        ["I already paid for premium account but why my account is still free?",
         "I was charged $10 instead of the student rate, very upset",
         "am struggling to add my son to Family account"]),
    "downloads_library": (
        "downloaded songs or saved library/playlists disappeared, offline mode, sync problems",
        ["All of my songs I had downloaded on my phone suddenly undownloaded",
         "my entire library is missing, playlists exist but no songs in my song section",
         "Any idea why your Android app occasionally wipes out every download?"]),
    "device_connect": (
        "playing on/with another device: Chromecast, Sonos, car, Alexa, console, smartwatch, Bluetooth, Spotify Connect",
        ["why does Spotify never work with chrome cast",
         "bluetooth controls apart from volume have stopped working since the iOS update",
         "can i download podcasts to a samsung gear sport and listen offline?"]),
    "content_missing": (
        "a song/album/artist/podcast is unavailable, greyed out, region-locked, wrong version/metadata, or a request to add content",
        ["why isn't the DVSN record available for streaming in Norway?",
         "where's Taylor Swift's new album?!",
         "the album is Pure Heroine - as in lady hero, not the drug"]),
    "feature_feedback": (
        "feature requests, UI/design complaints, recommendation/shuffle/playlist-algorithm complaints, praise, opinions",
        ["Why don't you include a 'sort by play count' option",
         "would be 100% perfect if it had a sleep timer on it",
         "you got a real solid team working for ya! Keep up the good work!"]),
    "other": (
        "spam, jokes, non-English, artist/label business, unclear, no support request",
        ["pull yourself together mate x",
         "check dm please",
         "what is the best playlist for studying"]),
}
INTENTS = list(TAXONOMY)

SYSTEM = "Classify a tweet sent to Spotify support into exactly one intent.\n\n" + "\n".join(
    f"{k}: {d}\n" + "".join(f'  - "{e}"\n' for e in ex) for k, (d, ex) in TAXONOMY.items()
) + ("\nIf two intents apply, pick the one the customer needs help with first.\n"
     'Reply with JSON only: {"intent": "<one of the names above>", "confidence": <0-1>, '
     '"rationale": "<under 12 words>"}')


def classify(text):
    raw = chat(SYSTEM, text[:600], json_mode=True, max_tokens=60)
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        out = {}
    intent = out.get("intent")
    if intent not in TAXONOMY:
        intent = "other"
    try:
        conf = min(1.0, max(0.0, float(out.get("confidence", 0))))
    except (TypeError, ValueError):
        conf = 0.0
    return {"intent": intent, "confidence": conf, "rationale": str(out.get("rationale", ""))[:200]}


if __name__ == "__main__":
    import time
    for t in ["I paid for premium and it still says free", "cant login, someone changed my email", "hola que tal"]:
        t0 = time.time()
        print(classify(t), f"{time.time()-t0:.2f}s")
    assert classify("I paid for premium and it still says free")["intent"] == "billing_subscription"
