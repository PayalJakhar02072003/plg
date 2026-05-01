"""CLI: compare two texts from stdin args or short demo strings."""

from __future__ import annotations

import argparse
import sys

from plagiarism.config import AggregationMode, DEFAULT_MODEL_NAME
from plagiarism.detector import SemanticPlagiarismDetector
from plagiarism.embedder import SentenceEmbedder


def main() -> None:
    p = argparse.ArgumentParser(description="Semantic similarity between two texts.")
    p.add_argument("text_a", nargs="?", default="", help="First document (or use --file-a)")
    p.add_argument("text_b", nargs="?", default="", help="Second document (or use --file-b)")
    p.add_argument("--file-a", type=str, default=None)
    p.add_argument("--file-b", type=str, default=None)
    p.add_argument(
        "--aggregation",
        choices=[m.value for m in AggregationMode],
        default=AggregationMode.MAX.value,
    )
    p.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    args = p.parse_args()

    a = args.text_a
    b = args.text_b
    if args.file_a:
        a = open(args.file_a, encoding="utf-8").read()
    if args.file_b:
        b = open(args.file_b, encoding="utf-8").read()
    if not a.strip() or not b.strip():
        print("Provide two non-empty texts or --file-a / --file-b.", file=sys.stderr)
        sys.exit(1)

    embedder = SentenceEmbedder(model_name=args.model)
    det = SemanticPlagiarismDetector(embedder, aggregation=args.aggregation)
    r = det.compare(a, b)
    print(f"Model: {r.model_name}")
    print(f"Aggregation: {r.aggregation}")
    print(f"Chunks: A={len(r.chunks_a)} B={len(r.chunks_b)} matrix={r.matrix_shape}")
    print(f"Score: {r.score:.4f} ({r.similarity_percent}% similarity index)")
    print(f"Verdict: {r.verdict_text}")
    print()
    print(r.disclaimer)


if __name__ == "__main__":
    main()
