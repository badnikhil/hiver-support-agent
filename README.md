# SpotifyCares support agent

## 1. What this is

An AI first-responder for the @SpotifyCares Twitter support account, built on the Kaggle "Customer Support on Twitter" dump. It reads one inbound customer tweet, classifies the intent, retrieves how the real brand answered similar tweets, drafts a reply in the brand's voice, and decides whether the reply can be sent automatically or must go to a human. Everything — classifier, drafter, judge — runs on a local `qwen2.5:3b` through Ollama; there is no cloud API in the loop.

Architecture (one message, ~2.5 s on a laptop GPU):

```
tweet ──► classify (qwen2.5:3b, JSON, 8 intents + confidence)
      ──► retrieve (TF-IDF word+char n-grams over 28k SpotifyCares threads → top-5 exemplars)
      ──► draft   (qwen2.5:3b, grounded-only prompt, exemplars as context; post-process drops
                   URLs not present in exemplars, trims to 280 chars)
      ──► policy  (deterministic rules over intent / text / retrieval score / draft →
                   auto | escalate, with a stated reason)
```

Why a local 3b model and no API: I had no API keys, and I wanted every number here reproducible on a reviewer's machine with nothing but `ollama pull`. The price is real: the 3b model is the main quality ceiling. The drafter paraphrases well when retrieval is good and invents steps when it is not; the judge barely discriminates (section 9).

## 2. Quickstart (reproduce in under 15 minutes)

Prerequisites: [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com) running on `localhost:11434`, and

```
ollama pull qwen2.5:3b          # ~2 GB, the only model needed
ollama pull qwen2.5:7b          # optional, better judge: JUDGE_MODEL=qwen2.5:7b make eval
```

Then:

```
make data          # download (177 MB, anonymous Kaggle API, no account) + build threads, corpus, TF-IDF index. ~1 min after download
make eval-quick    # 30 golden rows through the pipeline + all baselines, no judge. ~2 min cold
make eval          # full 220-row eval incl. LLM judge, IAA, judge-vs-human → data/eval/results.md
uv run -m agent.pipeline "my songs keep skipping on android"   # single-message demo
```

Two things make this fast. Every LLM call is cached in `data/cache/llm.sqlite`, keyed on model + full message body, so re-runs are free. And `data/eval/predictions.jsonl` (the pipeline's output on the golden set) plus `data/eval/judge_scores.json` are committed: `make eval` reproduces the headline tables from them in seconds, without a GPU. Delete a line from `predictions.jsonl` to recompute that row; delete the file to recompute everything (~8 min pipeline + ~11 min judge, measured on an RTX 2050 4 GB).

## 3. Problem framing

**Why SpotifyCares.** I profiled the 15 largest brands (`scripts/explore_brands.py`). Most are useless for an *answering* agent: telcos push 72–86% of first replies to "DM us your account", AmazonHelp's 0.6% DM rate hides a 47% "contact us here: link" rate, and airline problems cannot be answered from tweet text. SpotifyCares has 26,966 threads, 78% English, 31% DM redirects, and ~37% substantive first replies (troubleshooting steps, help-centre links, availability answers) about one product. That is the best "answerable from public knowledge" ratio in the dataset.

**What "good" means here.** Auto-handle how-to / troubleshooting / content-availability / feedback tweets with a reply grounded in what the brand actually says, in the brand's voice. Never auto-handle anything needing account access (billing, login), a legal/churn threat, PII, or a repeat contact. The dangerous error is an *unsafe auto*: a bot reply to someone who needed a human; over-escalating is the tolerable one. Metrics in priority order: unsafe-auto rate, escalation recall, reply quality on the auto slice, intent accuracy.

**What I chose not to build.** Multi-turn (the agent sees only the first customer tweet); embeddings or a vector DB; fine-tuning; a real DM/handoff integration; non-English handling (routed to `other`, escalated); RAG over support.spotify.com (help-centre links in exemplars are reused, never fetched); an LLM inside the escalation policy — rules only, on purpose (section 11).

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

Merges and splits, decided on 150 random corpus messages: `offline_download` became `downloads_library` because "my whole library disappeared" is as common as "downloads vanished" and has the same fix path. "Add this song" went into `content_missing`, not feedback, because the reply is the same ("we'll pass it to the label / not licensed in your region"). Praise stays in `feature_feedback` because it gets a real reply; `other` means no reply is needed. Rejected: a `usage_question` intent (how-tos are almost always about one of the areas above; the 3b would split hairs) and splitting billing into payment vs plan (first reply is "DM us your email" either way — no routing value).

## 5. Golden set

**Sampling** (`scripts/sample_golden.py`, seed 42, fully deterministic). Start from the 28,006 cleaned SpotifyCares threads, remove the 3,000-thread sample used for silver labels and the TF-IDF baseline, leaving 25,006. Draw 200 stratified uniformly by month (40 each from 2017-09, -10, -11, -12 and one pooled pre-September bucket) — the dump is 94% October–November 2017, and uniform-by-month deliberately over-represents the tails to catch drift in product and reply style rather than mirror traffic. Add 20 hard cases by regex family: 5 very short (<30 chars), 6 angry/legal (`cancel|refund|lawyer|hacked|scam|sue`), 5 with two issues ("also", "another problem"), 4 non-support chatter. Total 220.

**Labelling** (`data/golden/labeling_notes.md` has the full rule list). Each row was labelled from `customer_text` plus the full thread, following written guidelines. Key rules: label what the customer needs, judged from the whole thread — the brand's reply is a tiebreak, not ground truth; quote-tweets of Spotify promos (9 rows) get the intent of the real question in the next turn and are marked `hard`; "Premium paid but not active" is billing, not playback; hacked/Facebook/delete-account is login even when money is mentioned; "launch in India" is content_missing (region availability); app on a TV/console is device_connect, app on the phone is playback; algorithm complaints are feedback even when phrased as bugs; artist/label business and non-English are `other`. Escalation: any login/billing that needs the account looked at, `other`, repeat contact, PII, and anger only when it carries a churn/legal threat; plan/offer questions the brand answered publicly with a link stay auto even though the intent is billing; profanity without a threat stays auto.

Annotation was done with AI assistance (Claude Code agents acting as annotators, following the guidelines in labeling_notes.md); I reviewed the guidelines and the hard cases.

**Distribution.** feature_feedback 57, billing_subscription 39, content_missing 33, playback_issue 31, other 22, login_account 21, device_connect 9, downloads_library 8. Escalate = 81/220 (36.8%); 67 rows marked `hard`.

**Agreement.** A second, independent pass on a 60-row overlap (`golden_annotator_b.jsonl`): intent Cohen's κ = `{{iaa_intent_kappa}}` (raw `{{iaa_intent_raw}}`), should_escalate κ = `{{iaa_esc_kappa}}` (raw `{{iaa_esc_raw}}`). The labels are frozen; nothing in `golden.jsonl` was edited after the pipeline was run on it.

## 6. Evaluation harness

`eval/run_eval.py` produces `data/eval/results.md` and `results.json`. Automated metrics:

- **Intent**: accuracy, macro-F1, per-class P/R/F1, confusion matrix.
- **Escalation**: precision/recall/F1 for both classes, escalation share, and **unsafe autos** (predicted auto, human said escalate) — the number I treat as primary.
- **Reply quality**: LLM-as-judge, `qwen2.5:3b` at temperature 0, JSON output. Four dimensions plus overall, each 1–5 with anchors: *grounded* (5 = every step/claim/link appears in the reference reply or the exemplars; 1 = invents policy, steps or links), *helpful* (5 = addresses the actual issue with an actionable step; 1 = ignores it), *tone* (5 = warm, concise, under 280 chars, sounds like SpotifyCares; 1 = rude/robotic/too long), *safe* (5 = no promises of refunds/dates/features, never asks for passwords or card numbers publicly; 1 = does), *overall* (5 = send as-is; 3 = light edit; 1 = must not send). The judge sees the customer tweet, the candidate, the real brand reply as reference, and the retrieved exemplars. Reported as means, % overall ≥ 4, and a paired win/tie/loss against each baseline.
- **Judge-vs-human agreement**: `eval/sample_for_human.py` takes 60 seeded golden rows × 2 systems (agent draft, nearest-exemplar) = 120 items, shuffled and blind (system identity in a separate key file). I score them on the same rubric; `run_eval` then reports Spearman ρ and quadratic-weighted κ on `overall`, per-dimension ρ, and plain κ on the binary "acceptable" (≥ 4) decision.
- **Inter-annotator agreement** on the golden set as above.

Baselines: intent — majority class, a 7-regex keyword classifier, TF-IDF + logistic regression trained on 2,813 silver labels (produced by the same 3b classifier on a disjoint sample); escalation — always, never, keyword (`refund|cancel|charged|login|password|hacked|account`); replies — `reply_trivial` (one canned "DM us your email" reply) and `reply_nearest` (the top exemplar's real brand reply, verbatim).

## 7. Results

Intent (n = 220):

| system | accuracy | macro-F1 |
|---|---|---|
| majority (billing) | `{{int_majority_acc}}` | `{{int_majority_f1}}` |
| keyword | `{{int_keyword_acc}}` | `{{int_keyword_f1}}` |
| tfidf_lr (silver) | `{{int_tfidf_acc}}` | `{{int_tfidf_f1}}` |
| **agent (qwen2.5:3b)** | `{{int_agent_acc}}` | `{{int_agent_f1}}` |

Weakest classes: `other` (F1 `{{f1_other}}`) and `content_missing` (`{{f1_content}}`), which leak into each other and into `feature_feedback`. `downloads_library` and `login_account` are strong.

Escalation (n = 220, 81 gold escalates):

| system | esc P | esc R | unsafe autos | unsafe rate | esc share |
|---|---|---|---|---|---|
| always_escalate | 0.37 | 1.00 | 0 | 0 | 1.00 |
| never_escalate | – | 0 | 81 | 0.37 | 0 |
| keyword_escalate | `{{esc_kw_p}}` | `{{esc_kw_r}}` | `{{esc_kw_unsafe}}` | `{{esc_kw_unsafe_rate}}` | `{{esc_kw_share}}` |
| **agent** | `{{esc_agent_p}}` | `{{esc_agent_r}}` | `{{esc_agent_unsafe}}` | `{{esc_agent_unsafe_rate}}` | `{{esc_agent_share}}` |

First-rule escalation reasons for the agent: `{{esc_reasons}}`.

Reply quality (LLM judge, n = 220):

| system | grounded | helpful | tone | safe | overall | % overall ≥ 4 | win / tie / loss vs agent |
|---|---|---|---|---|---|---|---|
| reply_trivial | `{{j_triv_g}}` | `{{j_triv_h}}` | `{{j_triv_t}}` | `{{j_triv_s}}` | `{{j_triv_o}}` | `{{j_triv_pct}}` | `{{j_triv_wtl}}` |
| reply_nearest | `{{j_near_g}}` | `{{j_near_h}}` | `{{j_near_t}}` | `{{j_near_s}}` | `{{j_near_o}}` | `{{j_near_pct}}` | `{{j_near_wtl}}` |
| **agent** | `{{j_agent_g}}` | `{{j_agent_h}}` | `{{j_agent_t}}` | `{{j_agent_s}}` | `{{j_agent_o}}` | `{{j_agent_pct}}` | – |

Judge vs human (120 blind items): overall Spearman ρ = `{{jh_spearman}}`, quadratic-weighted κ = `{{jh_qwk}}`, κ on acceptable (≥ 4) = `{{jh_acc_kappa}}`; per-dimension ρ: `{{jh_dims}}`. Human means by system: `{{jh_human_means}}`.

Headline, in one sentence: the agent beats every baseline on intent and keeps unsafe autos to `{{esc_agent_unsafe}}`/220, but it does so by escalating `{{esc_agent_share}}` of traffic, and the judge's reply-quality numbers are too flat to separate the three reply systems with confidence.

## 8. Failure analysis

**F1. The real question is not in the tweet.** Nine golden rows have `customer_text` equal to a Spotify promo or status tweet the customer quote-tweeted; the actual question is in their next turn. Example `189825`: text is "Students now get access to Hulu with Spotify Premium. And it's still just $4.99/month" — the customer's real ask (military discount) is two turns later, in a 354-turn thread. The pipeline classifies the promo (billing, correct by luck) and drafts "Can you DM us your username?" Same for `100177` ("All clear! Everything's looking good again") where the real complaint is "your website is down". Hypothesis: single-turn input is the wrong unit; the agent needs the customer's first non-quoted turn, or the whole thread up to the first brand reply.

**F2. The drafter copies the exemplar's DM-ask even when a public answer exists.** `needs_account_access` fires on `{{n_needs_account}}`/220 drafts. Example `555492`: "upgraded to premium however am not able to skip songs?" — a known, publicly-answerable issue (Premium not applied to the right account), but the top exemplar asked for a DM, so the draft asks for a DM, so the policy escalates. In `pipeline_samples.md`, a "downloads keep disappearing" tweet whose real reply pointed to the "Downloads unexpectedly removed" help page got a DM-ask draft because exemplar #1 was a DM-ask. Hypothesis: 31% of the corpus's replies are DM redirects and the drafter treats the top exemplar as the template; the retrieval re-ranking that demotes bare redirects only catches short ones (<80 chars), and Spotify's DM-asks are ~120 chars. This one failure mode is why escalation precision is low.

**F3. Hallucinated steps when retrieval is weak.** When the top retrieval score is ≤ 0.30 the 3b model fills the gap. `511827` ("import an OPML feed of my podcasts?") → "import your OPML feed into iTunes first, then sync it with Spotify" — no exemplar says this and it is not true. `2021156` ("could I make playlist making a career") → "explore our job openings page". The sample run had "use the 'Play All' option if available, or select each track one by one" for a shuffle question. Hypothesis: the grounding instruction is not enforceable on a 3b model; the `MIN_RETRIEVAL = 0.25` gate is doing real work but is set too low. `{{fill from final predictions}}`: check whether the gate was raised and which of these became escalations.

**F4. `other` is the sink and the leak.** Gold `other` has F1 `{{f1_other}}`. Artist/label questions (`1910203` "as a verified artist why can't I create a playlist by that artist" → feature_feedback; `576201` artist-profile merge → login_account), image-only tweets (`554128` "please help!! <pic>" → content_missing), one-word tweets (`1806156` "account." → login_account) and non-English (`539199`, Filipino, about a failed Premium payment → billing_subscription — the English gate needs ≥ 6 Latin words and none of ~60 stopwords, and "premium" passed as English). Three of the six unsafe autos are gold-`other` tweets the classifier confidently gave a real intent and the drafter answered with an invented link. Hypothesis: the classifier has no "I don't know" — confidence is 0.8–1.0 on everything, so `low_confidence` never fires on a wrong-but-confident label. A cheap fix is a second check: "is there an explicit, English support request in this text? yes/no".

**F5. The judge barely judges.** The 3b judge collapses to a pattern (grounded 5, helpful ~1.5, tone 5, safe 5, overall 3) for most items; `helpful` averages `{{j_agent_h}}` for the agent and `{{j_near_h}}` for real brand replies. In the 3-item calibration test it did not recognise that `reply_nearest` *is* the first exemplar (grounded 1) until the prompt spelled it out. `{{jh_spearman}}` Spearman with human on `overall` is the only number that says whether it measures anything. Hypothesis: 3b models at temp 0 anchor on the rubric's middle value and on surface features (presence of a link → grounded); a 7b judge or pairwise rather than absolute scoring would help more than prompt tweaks.

## 9. What is misleading about my headline number?

- **The golden set was labelled by the same process that designed the taxonomy.** Definitions, edge rules and labels came from the same head (and the same assistants). A labeller with a different sense of "is a TV app playback or device_connect?" would move accuracy by several points. The overlap κ is high partly because both passes followed the same written rules.
- **Prompt and threshold changes were selected on the golden set.** The classifier prompt was iterated on a separate 150-row sample, but policy thresholds and the drafter's post-processing were adjusted after looking at golden output. There is no held-out set; treat the intent number as optimistic.
- **The "safe" number is bought by escalating most things.** `{{esc_agent_unsafe}}` unsafe autos looks good next to 81 for never-escalate, but the agent escalates `{{esc_agent_share}}` of traffic against a true rate of 37%. `always_escalate` has zero unsafe autos and is not a good agent. The honest comparison is unsafe-auto rate *at a given auto rate*, and there the agent is only modestly better than the keyword rule.
- **The judge is a 3b model that barely discriminates.** Means of ~3 with 60%+ ties say almost nothing about which system writes better replies. Only the judge-vs-human block is evidence, and it is 120 items from one human (me).
- **Month skew.** 94% of the data is Oct–Nov 2017. The golden set over-represents the tails by design, so it is not traffic-representative either way; iOS 11 / iPhone X era problems dominate.
- **Exemplars come from the same corpus as the reference replies.** The golden row's own thread is excluded from retrieval, but a near-duplicate tweet from the same week is not. A draft that copies a near-duplicate's reply scores as grounded and helpful — fine for a support bot, but it inflates the gap over a system facing genuinely new questions.
- **The reference "brand reply" is often itself bad.** Real replies asked for a DM on things needing no account (`47` duplicate in Discover Weekly, `218` add-a-song, `9` a joke), answered hacked-account reports with a public link (`207`, `208`), and once read as sarcastic (`12`). The judge treats the reference as ground truth for grounding, so a draft that improves on a bad reference can be penalised.
- **Single-turn only.** F1 shows at least 9/220 rows cannot be answered from the first tweet at all; the real product would have the thread.

## 10. What I would do with one more week

1. Score the 120 human items properly (two people, not one), and if judge-vs-human ρ is below ~0.5, switch the judge to `qwen2.5:7b` and pairwise "which is better" instead of absolute 1–5.
2. Feed the thread up to the first brand reply, not just the first tweet (fixes F1, likely helps F4).
3. Fix F2 at the source: re-rank retrieval to put substantive replies first using a better redirect detector (DM regex *and* no link/steps, any length), and tell the drafter explicitly which exemplars are redirects.
4. Add a "does this text contain an explicit English support request?" gate before classification, and raise `MIN_RETRIEVAL`; measure unsafe autos and auto rate together as a single curve.
5. A proper held-out split: re-sample 100 more golden rows, freeze, and report on those only.
6. Calibrate confidence (the 3b's 0.8–1.0 is useless); a cheap option is agreement between the LLM label and the TF-IDF+LR label as the confidence signal.
7. Try `bge-small` embeddings against TF-IDF on the retrieval step, measured by whether the draft's judge score moves, not by retrieval metrics.

## 11. Decision log

- **SpotifyCares, not AmazonHelp/AppleSupport.** Highest share of substantive public first replies (37%) on a single English-language product; the bigger brands are mostly redirects.
- **8 intents, not the 77 of Banking77-style taxonomies.** Reply routing only needs the fix-path granularity; the 3b classifier's accuracy falls off fast with more, similar classes.
- **Local `qwen2.5:3b`, no API.** No keys; every reviewer can reproduce; accepted that it caps quality and said so.
- **TF-IDF (word 1-2gram + char_wb 3-5gram), not embeddings.** Support tweets are lexically dense ("charged twice", "skipping", "greyed out"); TF-IDF hit 0.6–0.8 on near-duplicates and builds in 10 s with no extra model. Char n-grams handle typos.
- **Rules-only escalation policy, no LLM.** The unsafe-auto decision must be auditable and deterministic; every escalation carries the rule that fired. An LLM here would be a second uncalibrated 3b opinion.
- **Escalate when top retrieval score < 0.25.** The sample run showed the drafter starts inventing exactly where retrieval is weak; this threshold is the cheapest hallucination guard.
- **`needs_account_access` derived from the draft, not the tweet.** If the draft itself asks for a DM, sending it "automatically" is pointless; a human should take over. It also makes the policy robust to a wrong intent.
- **Silver labels (3b classifier on 2,813 disjoint rows) to train the TF-IDF+LR baseline.** Gives a "simple" baseline that is actually trained on something, without hand-labelling thousands of rows; the ceiling of that baseline is the 3b's own errors, and that is disclosed.
- **Stratified-by-month + 20 regex hard cases for the golden set.** Uniform random would have been 94% Oct–Nov 2017 billing/iOS-11 tweets; I wanted drift and the hard tail in the eval.
- **Second annotator pass on a 60-row overlap, not all 220.** Enough to compute κ; the marginal cost of the full set was better spent on scoring replies.
- **Judge rubric: grounded / helpful / tone / safe / overall.** Grounded and safe are the two things a support bot can get wrong in a way that hurts; tone is what the brand cares about; helpful is what the customer cares about; overall is the send/don't-send decision the human scores too.
- **Drop URLs not present in exemplars, post-hoc in Python.** The 3b model produced plausible-looking `t.co` links; a regex filter is 100% reliable where a prompt instruction was not.
- **English gate in Python, not in the prompt.** "Text not in English → other" was ignored by the 3b on all four test cases; a stopword heuristic flagged 616/28k corpus rows and spot-checked clean.
- **sqlite LLM cache keyed on sha256(model + messages + options).** Reruns are free; changing a prompt or model invalidates exactly what it should; the judge model is part of the key.
- **Commit `predictions.jsonl` and `judge_scores.json`.** Lets `make eval` reproduce every table in seconds and makes the eval an append-only, resumable log (delete a line to recompute a row).
- **Fixed `num_ctx=4096` everywhere and identical system prompts.** Changing `num_ctx` reloads the model (+5.7 s); identical prefixes get KV-cached, which is what makes 0.3 s classification possible.

## 12. Borrowed / cited

- Dataset: *Customer Support on Twitter*, Kaggle, user `thoughtvector` (Stuart Axelbrooke). Licence per the Kaggle page (CC BY-NC-SA 4.0 at the time of writing; see the Kaggle page for the authoritative text). Downloaded via the public `kaggle.com/api/v1/datasets/download` endpoint, no credentials.
- Banking77 (Casanueva et al., 2020) was *not* used; the taxonomy here is my own, derived from the SpotifyCares corpus.
- Model: Qwen2.5-3B-Instruct (Alibaba Cloud, Apache 2.0), served by [Ollama](https://ollama.com) as `qwen2.5:3b` (Q4_K_M).
- scikit-learn for TF-IDF, logistic regression, Cohen's κ, precision/recall; scipy for Spearman ρ; pandas/pyarrow for the data pipeline.
- The LLM-as-judge rubric follows the general pattern of G-Eval (Liu et al., 2023) and MT-Bench (Zheng et al., 2023): a fixed rubric with 1–5 anchors, absolute scoring plus a pairwise comparison, and an explicit judge-vs-human agreement check. The rubric text itself is mine.
- AI coding assistants (Claude Code) were used throughout for code, exploration, drafting this report, and — as stated in section 5 — as annotators under my written guidelines.
