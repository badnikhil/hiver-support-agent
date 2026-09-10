# hiver — SpotifyCares support agent (take-home)

## Layout
- `agent/` — pipeline: `intents.py` (classify), `retrieval.py` (TF-IDF), `reply.py` (draft + baselines), `policy.py` (rules), `llm.py` (Ollama + sqlite cache), `pipeline.py` (CLI / batch).
- `eval/` — `baselines.py`, `judge.py` (LLM judge), `run_eval.py` (-> `data/eval/results.md`), `sample_for_human.py`.
- `scripts/` — data prep: download, build_threads, build_corpus, build_index, sample_golden, silver_labels, explore_brands.
- `data/golden/golden.jsonl` (220 labelled, FROZEN), `golden_annotator_b.jsonl` (60 overlap), `labeling_notes.md`.
- `data/eval/predictions.jsonl` + `judge_scores.json` are committed so `make eval` reproduces tables without a GPU.
- `agent-docs/` — agent knowledge base (gitignored, local only). Read it before working; update it as you go.
- README.md is the report.

## Run
- `make data` / `make silver` / `make predict` / `make eval-quick` / `make eval` / `make all`.
- Demo: `uv run -m agent.pipeline "text"`. Self-checks: `uv run -m agent.policy|reply|retrieval|eval.baselines|eval.judge`.
- Ollama at `localhost:11434`, model `qwen2.5:3b` (`OLLAMA_MODEL`, `JUDGE_MODEL` env to override).

## Conventions
- `uv` only (`uv run ...`); Python 3.12. Terse commit messages, no trailers/attribution lines.
- Golden set is frozen: never edit labels to fit the model. Fix the model or note the disagreement in agent-docs.
- LLM cache: `data/cache/llm.sqlite`, keyed on sha256(model+messages+options). Gitignored. Reruns are free; a changed prompt re-runs.
- `predictions.jsonl` is append-only and resume-safe: delete a line to recompute that row. A policy-only change needs no re-predict: re-apply `policy.decide` to the cached rows (they carry text/intent/confidence/exemplars/draft).
- agent-docs/, scratch files, helper notes: never `git add`.
- Numbers in README come from `data/eval/results.md`; re-fill after any eval rerun.

## Gotchas
- Scripts in `scripts/` do `sys.path.insert(0, ROOT)` because `uv run scripts/x.py` does not put the repo root on the path.
- Keep `num_ctx` fixed (4096) across all calls: changing it reloads the model (+5.7 s per switch).
- Keep system prompts byte-identical across calls so Ollama KV-caches the prefix (0.3 s classify vs ~1 s).
- The 3b ignores "non-English -> other" in the prompt; `intents.english_ish` does it in Python. Don't move it back.
- Classifier confidence is uncalibrated (0.8–1.0 always); don't add thresholds on it.
- `reply.py` drops any URL not present in exemplars post-hoc; this is deliberate insurance, keep it.
- Judge (3b) collapses to grounded 5 / helpful 1 / tone 5 / safe 5 / overall 3; only judge-vs-human is real evidence.
- GPU: 4 GB VRAM; 7b judge offloads to CPU (~5–8 s/call). Run judge as a separate pass from drafts.
