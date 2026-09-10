"""Build the TF-IDF retrieval index. usage: uv run scripts/build_index.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.retrieval import INDEX, build_index

print(f"indexed {build_index()} threads -> {INDEX}")
