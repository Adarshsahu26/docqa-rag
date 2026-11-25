"""Bulk-load a folder of documents into the persisted index, without starting the server.

    python -m app.ingest sample_docs

Handy for first-time setup and demos; the API loads the same index when it starts.
Stop the server first if it is running: two processes should not write the index at once.
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from .config import Settings
from .errors import DocQAError
from .parsing import SUPPORTED_EXTENSIONS
from .rag import RagPipeline
from .vector_store import StoredDocument


def ingest_folder(pipeline: RagPipeline, folder: Path) -> tuple[list[StoredDocument], list[str]]:
    """Ingest every supported file directly inside `folder`.

    Returns the documents now in the index and one message per file that could not be read.
    Files that are already indexed are returned unchanged (ingestion is idempotent).
    """
    docs: list[StoredDocument] = []
    errors: list[str] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            docs.append(pipeline.ingest(path.name, path.read_bytes()))
        except DocQAError as exc:
            errors.append(f"{path.name}: {exc}")
    return docs, errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="folder containing .pdf, .docx, .txt or .md files")
    args = parser.parse_args(argv)
    if not args.folder.is_dir():
        parser.error(f"{args.folder} is not a folder")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()  # same .env the server reads, so both use the same settings and data folder
    pipeline = RagPipeline.from_settings(Settings.from_env())

    docs, errors = ingest_folder(pipeline, args.folder)
    for doc in docs:
        print(f"  {doc.doc_id}  {doc.num_chunks:>3} chunks  {doc.filename}")
    print(f"{len(docs)} file(s) indexed. The index now holds {len(pipeline.documents())} document(s) "
          f"and {len(pipeline.store)} chunks.")
    for message in errors:
        print(f"  skipped - {message}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
