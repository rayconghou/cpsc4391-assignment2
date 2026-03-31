from __future__ import annotations

import argparse
import json

from vectordb.core import VectorDB


def _vector_db_kwargs(ns: argparse.Namespace) -> dict:
    return {
        "random_seed": ns.seed,
        "num_tables": ns.lsh_tables,
        "num_hyperplanes": ns.lsh_bits,
        "max_features": ns.max_features,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vectordb",
        description="Local TF-IDF vector database with LSH + cosine rerank.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add", help="Insert a document (text) into the database.")
    p_add.add_argument("--db", required=True, help="Path to SQLite database file.")
    p_add.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for LSH hyperplanes (default: 0).",
    )
    p_add.add_argument(
        "--lsh-tables",
        type=int,
        default=10,
        help="Number of LSH hash tables (default: 10).",
    )
    p_add.add_argument(
        "--lsh-bits",
        type=int,
        default=16,
        help="Hyperplanes (bits) per LSH table (default: 16).",
    )
    p_add.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="TF-IDF max_features cap (default: 5000).",
    )
    p_add.add_argument(
        "--quiet",
        "--id-only",
        action="store_true",
        dest="id_only",
        help="Print only the new numeric document id.",
    )
    p_add.add_argument(
        "text",
        help="Text to store (quote if it contains spaces).",
    )

    p_q = sub.add_parser("query", help="Find the k most similar stored documents.")
    p_q.add_argument("--db", required=True, help="Path to SQLite database file.")
    p_q.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for LSH hyperplanes (default: 0).",
    )
    p_q.add_argument(
        "--lsh-tables",
        type=int,
        default=10,
        help="Number of LSH hash tables (default: 10).",
    )
    p_q.add_argument(
        "--lsh-bits",
        type=int,
        default=16,
        help="Hyperplanes (bits) per LSH table (default: 16).",
    )
    p_q.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="TF-IDF max_features cap (default: 5000).",
    )
    p_q.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )
    p_q.add_argument(
        "--json",
        action="store_true",
        help="Print results as JSON instead of human-readable text.",
    )
    p_q.add_argument("query_text", help="Query text.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "add":
        db = VectorDB(args.db, **_vector_db_kwargs(args))
        try:
            doc_id = db.add(args.text)
        finally:
            db.close()
        if args.id_only:
            print(doc_id)
        else:
            print(f"Added document id={doc_id}")
        return 0

    if args.command == "query":
        db = VectorDB(args.db, **_vector_db_kwargs(args))
        try:
            results = db.query(args.query_text, args.top_k)
        finally:
            db.close()
        if args.json:
            payload = [
                {"id": r.id, "text": r.text, "score": r.score} for r in results
            ]
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            for r in results:
                print(f"id={r.id}\tscore={r.score:.6f}\n{r.text}\n---")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
