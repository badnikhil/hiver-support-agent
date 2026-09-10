# SpotifyCares support agent

## 1. What this is

An AI first-responder for the @SpotifyCares Twitter support account, built on the Kaggle "Customer Support on Twitter" dump. It reads one inbound customer tweet, classifies the intent, retrieves how the real brand answered similar tweets, drafts a reply in the brand's voice, and decides whether the reply can be sent automatically or must go to a human. Everything — classifier, drafter, judge — runs on a local `qwen2.5:3b` through Ollama; no cloud API.

Architecture (one message, ~2.5 s on a laptop GPU):

```
tweet ──► classify (qwen2.5:3b, JSON, 8 intents + confidence)
      ──► retrieve (TF-IDF word+char n-grams over 28k SpotifyCares threads → top-5 exemplars)
      ──► draft   (qwen2.5:3b, grounded-only prompt, exemplars as context; post-process drops
                   URLs not present in exemplars, trims to 280 chars)
      ──► policy  (deterministic rules over intent / text / retrieval score / draft →
                   auto | escalate, with a stated reason)
```

Local and no API because I had no keys and wanted every number reproducible with nothing but `ollama pull`. The price: the 3b is the main quality ceiling — it invents steps where retrieval is weak, and as a judge it barely agrees with a human (section 7).

## 2. Quickstart

Prerequisites: [uv](https://docs.astral.sh/uv/), and — only for re-running the models — [Ollama](https://ollama.com) on `localhost:11434` with

```
ollama pull qwen2.5:3b          # ~2 GB, the only model needed
ollama pull qwen2.5:7b          # optional, better judge: JUDGE_MODEL=qwen2.5:7b make eval
```

Then:

```
make eval          # full 220-row eval: intent, escalation, judge, IAA, judge-vs-human → data/eval/results.md. 7 s from the committed files, no Ollama, no GPU
make eval-quick    # 30 golden rows + all baselines, no judge. 8 s from the committed files; ~2 min if the rows must be re-predicted
make data          # download (177 MB, anonymous Kaggle API) + threads, corpus, TF-IDF index. ~1 min after download. Needed for the demo and any re-prediction
uv run -m agent.pipeline "my songs keep skipping on android"   # single-message demo (needs Ollama + make data)
```

`data/eval/predictions.jsonl` (pipeline output on the golden set) and `data/eval/judge_scores.json` are committed and read back: the batch runner skips rows already predicted, and the judge reuses a stored score when the reply text is unchanged (default judge model only). Measured on a fresh shell: `make eval` 7 s, `make eval-quick` 8 s. Every LLM call is also cached in `data/cache/llm.sqlite` (gitignored), keyed on model + full message body, so local re-runs after a prompt change only pay for what changed. Delete a line from `predictions.jsonl` to recompute that row; delete the file to recompute everything (~8 min pipeline + ~11 min judge on an RTX 2050 4 GB).

## 3. Problem framing

**Why SpotifyCares.** Of the 15 largest brands (`scripts/explore_brands.py`), most are useless for an *answering* agent: telcos push 72–86% of first replies to "DM us", AmazonHelp 47% to "contact us here: link". SpotifyCares has 26,966 threads, 78% English, 31% DM redirects and ~37% substantive first replies about one product — the best "answerable from public knowledge" ratio in the dataset.

**What "good" means.** Auto-handle how-to / troubleshooting / content-availability / feedback tweets with a reply grounded in what the brand actually says, in its voice. Never auto-handle anything needing account access (billing, login), a legal/churn threat, PII, or a repeat contact. The dangerous error is an *unsafe auto* — a bot reply to someone who needed a human; over-escalating is the tolerable one. Metrics in priority order: unsafe-auto rate, escalation recall, reply quality on the auto slice, intent accuracy.

**Not built.** Multi-turn (the agent sees only the first customer tweet); embeddings; fine-tuning; a DM/handoff integration; non-English handling (routed to `other`, escalated); RAG over support.spotify.com (help-centre links in exemplars are reused, never fetched); an LLM inside the escalation policy — rules only, on purpose.

## 4. Intents

| intent | definition |
|---|---|
| playback_issue | app on the user's own device crashes, won't play, skips, buffers, high CPU, ads misbehaving |
| login_account | can't log in, hacked, password/email reset, Facebook link, delete/merge account |
| billing_subscription | wrong charge, refund, cancel, paid-but-still-Free, trial/student/family/duo plans, payment methods |
| downloads_library | downloads or saved library/playlists vanished, offline mode, sync |
| device_connect | a *second* device: Chromecast, Sonos, car, Alexa, TV, console, Bluetooth, Spotify Connect |
| content_missing | song/album/podcast unavailable, greyed out, region-locked, wrong metadata, "please add X", "launch in my country" |
| feature_feedback | feature requests, UI and algorithm complaints (shuffle, Discover Weekly), praise |
| other | non-English, spam, jokes, artist/label business, image-only, no request |

Decided on 150 random corpus messages; the rule was "split only where the first reply differs". So `offline_download` merged into `downloads_library`, "add this song" is `content_missing`, praise stays in `feature_feedback` (it gets a real reply), and a `usage_question` intent and a payment-vs-plan split were rejected.

## 5. Golden set

**Sampling** (`scripts/sample_golden.py`, seed 42). From 28,006 cleaned threads, remove the 3,000 used for silver labels. Draw 200 stratified uniformly by month (40 each from 2017-09, -10, -11, -12 and one pooled pre-September bucket) — the dump is 94% October–November 2017, and uniform-by-month deliberately over-represents the tails to catch drift. Add 20 hard cases by regex: 5 very short, 6 angry/legal (`cancel|refund|lawyer|hacked|scam|sue`), 5 with two issues, 4 non-support chatter. Total 220.

**Labelling** (full rules in `data/golden/labeling_notes.md`). Each row was labelled from `customer_text` plus the full thread: label what the customer needs — the brand's reply is a tiebreak, not ground truth; quote-tweets of Spotify promos (9 rows) get the intent of the real question in the next turn and are marked `hard`; artist/label business and non-English are `other`. Escalate: login/billing needing the account looked at, `other`, repeat contact, PII, anger only with a churn/legal threat; plan questions the brand answered publicly and profanity without a threat stay auto. Annotation used AI assistance (Claude Code agents as annotators under those rules); I reviewed the guidelines and hard cases.

**Distribution.** feature_feedback 57, billing_subscription 39, content_missing 33, playback_issue 31, other 22, login_account 21, device_connect 9, downloads_library 8. Escalate = 81/220 (36.8%); 67 rows `hard`.

**Agreement.** A second, independent pass on a 60-row overlap (`golden_annotator_b.jsonl`): intent Cohen's κ = 0.899 (raw 0.917), should_escalate κ = 0.927 (raw 0.967). Labels are frozen; nothing was edited after the pipeline ran on them.

## 6. Evaluation harness

`eval/run_eval.py` produces `data/eval/results.md` and `results.json`:

- **Intent**: accuracy, macro-F1, per-class P/R/F1, confusion matrix.
- **Escalation**: P/R/F1 for both classes, escalation share, and **unsafe autos** (predicted auto, human said escalate) — the primary number.
- **Reply quality, human-rated (primary)**: `eval/sample_for_human.py` takes 60 seeded golden rows × 2 systems (agent draft, nearest-exemplar real reply) = 120 items, shuffled and blind (system identity in a separate key file). One rater scored them on the rubric below, seeing only the tweet and the reply. Reported as per-system means, % overall ≥ 4, and a paired agent-vs-nearest win/tie/loss on the 60 shared rows.
- **Reply quality, LLM judge (secondary)**: `qwen2.5:3b`, temperature 0, JSON, 220 rows × 3 systems. Dimensions 1–5 with anchors: *grounded* (every step/claim/link is in the reference or exemplars vs invented), *helpful* (addresses the actual issue with a next step vs ignores it), *tone* (SpotifyCares voice, ≤ 280 chars vs rude/robotic/long), *safe* (no promised refunds/dates/features, no password/card asks), *overall* (5 send as-is / 3 light edit / 1 must not send). The judge also sees the real brand reply and the exemplars. Reported as means, % overall ≥ 4, paired win/tie/loss vs each baseline.
- **Judge-vs-human agreement** on the 120 items: Spearman ρ and quadratic-weighted κ on `overall`, exact and ±1 agreement, κ on the binary "acceptable" (≥ 4) decision, per-dimension ρ.
- **Inter-annotator agreement** on the golden set.

Baselines: intent — majority class, a 7-regex keyword classifier, TF-IDF + logistic regression trained on 2,813 silver labels (the same 3b classifier on a disjoint sample); escalation — always, never, keyword (`refund|cancel|charged|login|password|hacked|account`); replies — `reply_trivial` (one canned "DM us your email") and `reply_nearest` (the top exemplar's real brand reply, verbatim).

## 7. Results

Intent (n = 220):

| system | accuracy | macro-F1 |
|---|---|---|
| majority (billing) | 0.177 | 0.038 |
| keyword | 0.482 | 0.524 |
| tfidf_lr (silver) | 0.600 | 0.566 |
| **agent (qwen2.5:3b)** | **0.664** | **0.690** |

Weakest classes: `other` (F1 0.34) and `content_missing` (0.56), which leak into each other and into `feature_feedback`. `downloads_library` (1.00), `login_account` and `playback_issue` (0.77) are strong.

Escalation (n = 220, 81 gold escalates):

| system | esc P | esc R | unsafe autos | unsafe rate | esc share |
|---|---|---|---|---|---|
| always_escalate | 0.37 | 1.00 | 0 | 0 | 1.00 |
| never_escalate | – | 0 | 81 | 0.37 | 0 |
| keyword_escalate | 0.84 | 0.38 | 50 | 0.23 | 0.17 |
| **agent** | 0.43 | 0.94 | **5** | **0.023** | 0.81 |

First-rule reasons for the agent: account_intent 78, no_similar_history 57, low_confidence 24, needs_account_access 15, anger_legal_churn 4.

**Reply quality — human-rated (primary).** 120 blind items, 60 rows × {agent draft, nearest real brand reply}. The rater saw only the tweet and the reply, so `grounded` here means "nothing invented".

| system | grounded | helpful | tone | safe | overall | % overall ≥ 4 |
|---|---|---|---|---|---|---|
| reply_nearest | 4.17 | 2.60 | 4.05 | 4.87 | 2.80 | 33.3 |
| **agent** | 4.03 | 2.90 | 3.57 | 4.75 | 2.85 | 28.3 |

Paired on the 60 shared rows, agent vs nearest on `overall`: **21 wins / 21 ties / 18 losses — win rate 0.35, loss rate 0.30.** The agent is indistinguishable from pasting the nearest real brand reply: a little more helpful (it addresses this tweet rather than a neighbour's), clearly worse on tone (longer, stiffer, once with a stray agent signature) and on grounding (10/60 drafts scored grounded ≤ 2 vs 3/60 real replies). Mean overall 2.85 means the typical draft needs more than a light edit; 28% are sendable as-is. Worse: on the 17 human-scored drafts the policy would have *auto-sent*, human overall is 2.71, no better than the escalated ones (2.91) — the policy is not selecting the good drafts.

**Reply quality — LLM judge (secondary), n = 220:**

| system | grounded | helpful | tone | safe | overall | % overall ≥ 4 | win / tie / loss vs agent |
|---|---|---|---|---|---|---|---|
| reply_trivial | 3.79 | 1.35 | 4.98 | 4.92 | 3.00 | 15.9 | 25 / 127 / 68 |
| reply_nearest | 4.65 | 1.25 | 4.98 | 4.84 | 2.97 | 10.9 | 30 / 114 / 76 |
| **agent** | 4.37 | 1.74 | 4.96 | 4.89 | 3.20 | 35.0 | – |

**Judge vs human (120 items).** Spearman ρ on `overall` = **0.13**, quadratic-weighted κ = **0.10**, exact agreement 22.5%, within ±1 74.2%, κ on the binary "acceptable" decision = **0.04**. Per-dimension ρ: grounded 0.25, helpful 0.21, tone 0.06, safe 0.00. That is no agreement: ±1 is what two raters get by both sitting near 3; κ 0.04 on "sendable" is chance. The judge's ordering agent > trivial > nearest (35 / 16 / 11 % ≥ 4) is not confirmed by the human, who put nearest slightly *ahead* on % ≥ 4 (33 vs 28); that the judge's paired win rate vs nearest (0.345) lands on the human's (0.35) is coincidence at ρ = 0.13. The judge table is the 3b's opinion, not evidence; the human table is the result. Part of the gap is construct (the human never saw the reference, so human `grounded` is "nothing invented", judge `grounded` is "matches the reference"), but tone and safe have no such excuse.

**7b judge check.** `JUDGE_MODEL=qwen2.5:7b` on the same seeded 20 rows (60 calls): 7b un-collapses `helpful` (distribution {1:16, 2:25, 3:14, 4:4, 5:1} vs the 3b's {1:48, 3:11, 4:1}) and stops rubber-stamping grounded = 5 (agent 3.45 vs 4.6), but `overall` is still 3 on 42/60 items and paired ties stay at 60–65%, at 3× the cost (3.8 s/call, offloaded to CPU on 4 GB VRAM). 3b stays the default; 7b is the documented opt-in.

**Headline, in one sentence:** the agent beats every baseline on intent (0.66 vs 0.60) and keeps unsafe autos to 5/220, but only by escalating 81% of traffic; on reply quality a blind human rates it level with pasting the nearest real reply (win 0.35 / loss 0.30, overall 2.85 vs 2.80), and the 3b judge agrees with that human at ρ = 0.13, so its numbers are not evidence of anything.

## 8. Failure analysis

Examples are the final drafts in `data/eval/predictions.jsonl`; ids are `customer_tweet_id`.

**F1. The real question is not in the tweet.** 9/220 rows have `customer_text` equal to a Spotify promo or status tweet the customer quote-tweeted; the question is in their next turn. `189825`: "Students now get access to Hulu with Spotify Premium. And it's still just $4.99/month" → "Hey there! What exactly is going on with your Spotify? We'll see what we can suggest 🔍" — the real ask (military discount) is two turns later in a 354-turn thread; escalated as billing by luck. `100177`: Spotify's own "All clear! Everything's looking good again" → "We're glad you're all set now… give us a DM for further assistance" while the complaint is "your website is down". Hypothesis: single-turn input is the wrong unit; feed the thread up to the first brand reply.

**F2. The drafter fills in the exemplar's template instead of answering the problem.** The human rated `helpful` ≤ 2 on 21/60 agent drafts, mostly this. `778441`: "The app keeps creating new playlists labelled 'tracks' with only one song in each one. How do I stop this?" → "To troubleshoot, can you please send a screenshot of the issue on your device?" `2065778`: a customer clarifying an album name ("Pure Heroine - as in lady hero, not the drug") → "For 'Pure Heroine' by Lady Leshurr, can you please try restarting your device?" — wrong artist, restart template. `765659`: "someone has been using my account… I get emails in different languages" → "Could you please send the Song Link so we can look into it further?" `555492`: upgraded-but-can't-skip → "Are you getting any error messages when trying to skip songs…? does logging out and back in help?… rdw1981" — an agent's sign-off copied from the exemplar. 73/220 drafts still ask for a DM (`needs_account_access` fires on 71; after re-ranking redirects it is the *first* rule on only 15, down from 29). Hypothesis: the 3b treats the top exemplar as fill-in-the-blanks and substitutes the customer's nouns into it — that is where "Lady Leshurr" and "Song Link" come from. It needs to know which exemplars are redirects, or a "what is the customer actually asking" step first.

**F3. Hallucinated steps and links when retrieval is weak.** `511827`: "is there a way to import an OPML feed of my preferred podcasts to Spotify?" → "You can try importing your podcast feeds using the 'Add Podcasts' feature by following these steps: 1. Go to 2. Click on 'Import RSS Feed' 3. Paste your OPML feed URL" — the feature does not exist; the blank after "Go to" is the URL filter removing an invented link. `2021156`: "could I make playlist making a career" → "explore our current job openings for roles like Playlist Creator. Check out this link for more info:". `571379`: PayPal→Visa, $9.99→$10.14 → "Changing your payment method can sometimes cause temporary billing issues" — a fabricated explanation for what is tax. All three have top retrieval 0.24–0.27 and were escalated by `no_similar_history` once the gate went from 0.25 to 0.28: the policy caught them, the drafter did not. Of the 10/60 human-scored drafts with grounded ≤ 2, 7 were escalated; 3 went auto, e.g. `365597` ("any news on an iwatch app?") → "consider using third-party apps like 'Spotify for Apple Watch'", top 0.32, human grounded 1 / overall 1. Hypothesis: the grounding instruction is not enforceable on a 3b; the retrieval gate is the only real guard and it is blunt (81/220 rows sit under it, most answerable).

**F4. `other` is the sink and the leak.** Gold `other` F1 is 0.34, and **4 of the 5 unsafe autos are gold-`other` rows** the classifier confidently gave a real intent: `1910203` (verified artist can't create artist playlists → feature_feedback, auto, "Our Artist Support team would be the best resource… here: <link>"), `554128` ("please help!! <pic>" → content_missing, auto, "How long has this been happening? Can you try restarting the app"), `822279` ("hi! I need help!" → playback_issue, auto), `570525` (sarcastic "where can i find normal amounts of songs?" → content_missing, auto, "content gets temporarily removed… due to licensing changes"). Non-English leaks too: `539199` (Filipino, failed Premium payment → billing; the English gate needs ≥ 6 Latin words and no stopwords, and "premium" passed). Hypothesis: the classifier has no "I don't know" — confidence is 0.8–1.0 on everything, so `low_confidence` never fires on a wrong-but-confident label. Two prompt fixes were tried and reverted (section 9); next is a separate yes/no gate: "is there an explicit, English support request in this text?".

**F5. The judge barely judges.** The 3b judge collapses to grounded 5 / helpful 1 / tone 5 / safe 5 / overall 3 for most items: `helpful` averages 1.74 for the agent and 1.25 for real brand replies where the human gave 2.90 and 2.60; tone and safe are ~4.9 for everything. Judge-vs-human ρ on overall is 0.13, κ on "sendable" 0.04, per-dimension ρ never above 0.25. In the 3-item calibration test it did not recognise that `reply_nearest` *is* the first exemplar until the prompt spelled it out. Hypothesis: a 3b at temperature 0 anchors on the rubric's middle value and on surface features (a link present → grounded); 7b un-collapses `helpful` but still ties 60%+; pairwise scoring would help more than prompt tweaks.

## 9. What is misleading about my headline number?

- **The golden set was labelled by the same process that designed the taxonomy.** Definitions, edge rules and labels came from the same head (and the same assistants). A labeller with a different sense of "is a TV app playback or device_connect?" would move accuracy by several points; the overlap κ is high partly because both passes followed the same written rules.
- **Thresholds and prompts were selected on the golden set.** `MIN_RETRIEVAL` was raised 0.25 → 0.28 by sweeping the golden aggregate — the only setting that met "unsafe autos ≤ 6"; it cost 0.013 escalation F1 and +4 points of escalation share. Two classifier prompt variants (a sharper `other` definition; one few-shot per weak class) were run on the golden aggregate and reverted because macro-F1 fell (0.64 and 0.66 vs 0.69). No individual rows were inspected while tuning, but there is no held-out split; every agent number is optimistic.
- **"Safe" is bought by escalating 81% of traffic, and the residual is a classifier error.** 5 unsafe autos looks good next to 81 for never-escalate, but the agent escalates 81% against a true rate of 37%; `always_escalate` has zero unsafe autos and is not a product. 4 of the 5 that got through are gold-`other` rows misclassified as a real intent — the safety number is mostly the `other` class's recall (0.36) in disguise.
- **The auto slice is not the good slice.** Human-rated overall on drafts the policy would auto-send is 2.71, below the escalated ones (2.91). The policy is a retrieval-score and intent gate, not a quality gate.
- **The judge does not measure reply quality.** ρ = 0.13 with a human, κ 0.04 on "sendable". Ignore the judge's agent-beats-baselines table; the human table is one rater on 120 items and says "level with the nearest real reply".
- **Month skew; same-corpus exemplars.** 94% of the data is Oct–Nov 2017 and the golden set over-represents the tails by design, so it is not traffic-representative either way. The row's own thread is excluded from retrieval, but a near-duplicate from the same week is not; copying its reply looks grounded, which inflates the gap over genuinely new questions.
- **The reference "brand reply" is often itself bad.** Real replies asked for a DM on things needing no account (`47`, `218`, `9`), answered hacked accounts with a public link (`207`, `208`), once read as sarcastic (`12`). The human rated `reply_nearest` at overall 2.80 — the bar the agent is level with is low.

## 10. What I would do with one more week

1. A second human rater on the 120 items (one rater means the 0.35 / 0.30 paired result has no agreement number behind it), and switch the judge to pairwise "which is better" on `qwen2.5:7b` — ρ = 0.13 says absolute 1–5 from a 3b is not worth running.
2. Feed the thread up to the first brand reply, not just the first tweet (fixes F1, likely helps F4).
3. Fix F2 at the source: label each exemplar as redirect / template / substantive before drafting, and add a "restate the customer's ask in one line" step.
4. An "explicit English support request? yes/no" gate before classification (F4), and report unsafe autos and auto rate together as a curve, not one point.
5. A held-out split: re-sample 100 more golden rows, freeze, report on those only.
6. Calibrate confidence (the 3b's 0.8–1.0 is useless); cheapest option is agreement between the LLM label and the TF-IDF+LR label.
7. Try `bge-small` embeddings against TF-IDF on retrieval, measured by whether human-rated draft quality moves.

## 11. Decision log

- **SpotifyCares, not AmazonHelp/AppleSupport.** Highest share of substantive public first replies (37%) on one English-language product; the bigger brands are mostly redirects.
- **8 intents, not Banking77-style 77.** Reply routing only needs fix-path granularity; the 3b's accuracy falls off fast with more, similar classes.
- **Local `qwen2.5:3b`, no API.** No keys; every reviewer can reproduce; accepted that it caps quality.
- **TF-IDF (word 1-2gram + char_wb 3-5gram), not embeddings.** Support tweets are lexically dense; TF-IDF hits 0.6–0.8 on near-duplicates, builds in 10 s, and char n-grams handle typos.
- **Rules-only escalation policy.** The unsafe-auto decision must be auditable and deterministic; every escalation carries the rule that fired. An LLM here would be a second uncalibrated 3b opinion.
- **Escalate when top retrieval score < 0.28 (was 0.25).** The drafter invents exactly where retrieval is weak; the golden-aggregate sweep showed 0.28 was the only lever that held unsafe autos ≤ 6 after the drafter stopped copying DM-asks.
- **Policy changes re-apply `decide` to cached predictions; no re-predict.** Cached rows carry text / intent / confidence / exemplars / draft, so a threshold sweep is instant and holds the drafts fixed — it measures the policy and nothing else.
- **`needs_account_access` derived from the draft, not the tweet.** If the draft itself asks for a DM, sending it "automatically" is pointless; it also makes the policy robust to a wrong intent.
- **Silver labels (3b classifier on 2,813 disjoint rows) to train the TF-IDF+LR baseline.** A "simple" baseline actually trained on something, without hand-labelling thousands of rows; its ceiling is the 3b's own errors, disclosed.
- **Stratified-by-month + 20 regex hard cases; second annotator on a 60-row overlap only.** Uniform random would have been 94% Oct–Nov 2017 billing/iOS-11 tweets. 60 rows is enough for κ; the effort went into scoring replies.
- **Judge rubric: grounded / helpful / tone / safe / overall.** Grounded and safe are what hurts when a bot gets it wrong; tone is the brand's concern, helpful the customer's; overall is the send/don't-send decision.
- **Judge stays `qwen2.5:3b`; 7b is opt-in.** On the same 20 rows 7b un-collapsed `helpful` and stopped rubber-stamping grounded, but overall was still 3 on 42/60 and paired ties stayed 60–65%: 3× the cost for no better separation. `JUDGE_MODEL=qwen2.5:7b` is documented.
- **Hard rules in Python, not in the prompt.** URLs not in exemplars are dropped by regex (the 3b produced plausible `t.co` links despite the instruction); "non-English → other" is a stopword heuristic (ignored by the 3b on all four test cases; flagged 616/28k corpus rows, spot-checked clean).
- **Cache everything, commit the outputs.** sqlite LLM cache keyed on sha256(model + messages + options), so a changed prompt or model invalidates exactly what it should; `predictions.jsonl` and `judge_scores.json` are committed and read back, so `make eval` is a resumable, append-only log that reproduces every table in seconds.
- **Fixed `num_ctx=4096` and identical system prompts everywhere.** Changing `num_ctx` reloads the model (+5.7 s); identical prefixes get KV-cached, which is what makes 0.3 s classification possible.

## 12. Borrowed / cited

- Dataset: *Customer Support on Twitter*, Kaggle, user `thoughtvector` (Stuart Axelbrooke), CC BY-NC-SA 4.0 at the time of writing. Downloaded via the public `kaggle.com/api/v1/datasets/download` endpoint, no credentials.
- Banking77 (Casanueva et al., 2020) was *not* used; the taxonomy is my own, derived from the SpotifyCares corpus.
- Models: Qwen2.5-3B/7B-Instruct (Alibaba Cloud, Apache 2.0), served by [Ollama](https://ollama.com) as `qwen2.5:3b` / `qwen2.5:7b` (Q4_K_M).
- scikit-learn for TF-IDF, logistic regression, Cohen's κ, precision/recall; scipy for Spearman ρ; pandas/pyarrow for the data pipeline.
- The LLM-as-judge rubric follows the general pattern of G-Eval (Liu et al., 2023) and MT-Bench (Zheng et al., 2023): a fixed rubric with 1–5 anchors, absolute plus pairwise scoring, and an explicit judge-vs-human check. The rubric text is mine.
- AI coding assistants (Claude Code) were used throughout for code, exploration, drafting this report, and — as stated in section 5 — as annotators under my written guidelines.
