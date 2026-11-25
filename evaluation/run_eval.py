"""Measure retrieval quality on the bundled question set.

    python -m evaluation.run_eval                       # default embedder, three chunk sizes
    python -m evaluation.run_eval --embedder hashing    # offline lexical baseline
    python -m evaluation.run_eval --embedder sentence-transformers --show-misses

For every question the script asks the retriever for the top-k chunks. A question is
"found at rank r" when the r-th chunk comes from the right file AND contains the evidence
phrase that answers it. From those ranks it reports:

    hit@k  share of questions found within the first k chunks (higher is better)
    MRR    mean reciprocal rank, 1/rank averaged over questions (1.0 = always rank 1)

Retrieval is measured on its own, without a language model, so the numbers are cheap,
repeatable and isolate the part that chunking and embeddings control.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings
from app.embeddings import Embedder, build_embedder
from app.llm import ExtractiveLLM
from app.parsing import SUPPORTED_EXTENSIONS
from app.rag import Hit, RagPipeline
from app.vector_store import VectorStore

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = ROOT / "sample_docs"
DEFAULT_QUESTIONS = ROOT / "evaluation" / "questions.jsonl"


@dataclass
class Result:
    embedder: str
    chunk_size: int
    overlap: int
    chunks: int
    hit_at: dict[int, float]
    mrr: float
    misses: list[tuple[str, int | None]] = field(default_factory=list)  # (question, rank or None)


def load_questions(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for number, row in enumerate(rows, start=1):
        missing = {"question", "evidence"} - row.keys()
        if missing:
            raise ValueError(f"{path} entry {number} is missing {sorted(missing)}")
    return rows


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def first_hit_rank(hits: list[Hit], evidence: str, source: str | None) -> int | None:
    """1-based rank of the first chunk that is from `source` and contains `evidence`."""
    wanted = normalise(evidence)
    for rank, hit in enumerate(hits, start=1):
        if source and hit.record.filename != source:
            continue
        if wanted in normalise(hit.record.text):
            return rank
    return None


def build_pipeline(embedder: Embedder, corpus_dir: Path, chunk_size: int, overlap: int, top_k: int) -> RagPipeline:
    settings = Settings(
        data_dir=None,  # in memory: evaluation never touches the real index
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        top_k=top_k,
        min_score=-1.0,  # no relevance cut-off: we want to see the raw ranking
        llm_provider="extractive",
    )
    pipeline = RagPipeline(settings, embedder, VectorStore(embedder.dim, embedder.name), ExtractiveLLM())
    for path in sorted(corpus_dir.iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            pipeline.ingest(path.name, path.read_bytes())
    return pipeline


def evaluate(
    embedder: Embedder,
    corpus_dir: Path,
    questions: list[dict],
    chunk_size: int,
    overlap: int,
    ks: tuple[int, ...] = (1, 3, 5),
) -> Result:
    pipeline = build_pipeline(embedder, corpus_dir, chunk_size, overlap, top_k=max(ks))
    ranks: list[int | None] = []
    misses: list[tuple[str, int | None]] = []
    for row in questions:
        hits = pipeline.search(row["question"], top_k=max(ks))
        rank = first_hit_rank(hits, row["evidence"], row.get("source"))
        ranks.append(rank)
        if rank != 1:
            misses.append((row["question"], rank))
    total = len(questions)
    return Result(
        embedder=embedder.name,
        chunk_size=chunk_size,
        overlap=overlap,
        chunks=len(pipeline.store),
        hit_at={k: sum(1 for r in ranks if r is not None and r <= k) / total for k in ks},
        mrr=sum(1 / r for r in ranks if r) / total,
        misses=misses,
    )


def format_table(results: list[Result], ks: tuple[int, ...]) -> str:
    header = ["embedder", "chunk size (words)", "overlap", "chunks", *[f"hit@{k}" for k in ks], "MRR"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in results:
        cells = [r.embedder, str(r.chunk_size), str(r.overlap), str(r.chunks)]
        cells += [f"{r.hit_at[k]:.2f}" for k in ks] + [f"{r.mrr:.2f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedder", choices=["auto", "hashing", "sentence-transformers"], default="auto")
    parser.add_argument("--model", default=Settings().embedding_model, help="Sentence-Transformers model name")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[60, 120, 180], help="max words per chunk")
    parser.add_argument("--overlap-ratio", type=float, default=0.15, help="overlap as a share of the chunk size")
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5], help="the k values for hit@k")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--show-misses", action="store_true", help="list questions not answered at rank 1")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    embedder = build_embedder(args.embedder, args.model)  # built once: loading a model is slow
    ks = tuple(sorted(set(args.top_k)))
    results = [
        evaluate(embedder, args.corpus, questions, size, int(size * args.overlap_ratio), ks)
        for size in args.chunk_sizes
    ]

    print(f"{len(questions)} questions, corpus: {args.corpus}\n")
    print(format_table(results, ks))
    if args.show_misses:
        for r in results:
            print(f"\nNot found at rank 1 (chunk size {r.chunk_size}):")
            for question, rank in r.misses:
                print(f"  - rank {rank if rank else '>' + str(max(ks))}: {question}")


if __name__ == "__main__":
    main()
