.PHONY: data silver predict eval eval-quick all

data:
	uv run scripts/download_data.py
	uv run scripts/build_threads.py --brand SpotifyCares
	uv run scripts/build_corpus.py
	uv run scripts/build_index.py

silver:
	uv run scripts/sample_golden.py
	uv run scripts/silver_labels.py

predict:
	uv run -m agent.pipeline --jsonl data/golden/golden.jsonl --out data/eval/predictions.jsonl

eval:
	uv run -m eval.run_eval

eval-quick:
	uv run -m eval.run_eval --quick --skip-judge

all: data silver predict eval
