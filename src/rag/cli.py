"""Command-line interface: ``rag ingest <paths...>`` and ``rag ask "<question>"``.

Uses argparse (stdlib) to avoid an extra dependency. Both commands drive the
same ``RagPipeline`` as the API, backed by the persistent Chroma store so an
ingest in one invocation is visible to an ask in the next.
"""

from __future__ import annotations

import argparse
import sys

from .config import get_settings
from .generate import format_citation
from .logging_utils import configure_logging
from .pipeline import RagPipeline, build_pipeline


def _build(store_kind: str = "chroma") -> RagPipeline:
    settings = get_settings()
    configure_logging(settings.log_level)
    return build_pipeline(settings, store_kind=store_kind)


def _cmd_ingest(args: argparse.Namespace) -> int:
    pipeline = _build(store_kind=args.store)
    result = pipeline.ingest(args.paths)
    print(f"Ingested {result.documents} document(s) into {result.chunks} chunk(s).")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    pipeline = _build(store_kind=args.store)
    resp = pipeline.ask(args.question)

    print(resp.answer)
    if resp.grounded and resp.citations:
        print("\nCitations:")
        for c in resp.citations:
            print(f"  {format_citation(c)} (score={c.score:.3f})")
    elif not resp.grounded:
        print("\n(No sufficiently relevant context was found.)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag", description="Retrieval-Augmented Generation demo CLI."
    )
    parser.add_argument(
        "--store",
        choices=["chroma", "memory"],
        default="chroma",
        help="Vector store backend (default: chroma).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest files or directories.")
    p_ingest.add_argument("paths", nargs="+", help="Files/directories to ingest.")
    p_ingest.set_defaults(func=_cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a question against ingested documents.")
    p_ask.add_argument("question", help="The natural-language question.")
    p_ask.set_defaults(func=_cmd_ask)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
