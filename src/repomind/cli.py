"""CLI for snapshot import, evidence inspection and hybrid retrieval."""

import argparse
import logging
import os
from pathlib import Path

from repomind.ingestion import GitImportError, SourceError
from repomind.service import RepoMindService
from repomind.retrieval.embedding import DashScopeEmbedding, EmbeddingError
from repomind.retrieval.search import HybridSearch


def main() -> int:
    parser = argparse.ArgumentParser(prog="repomind")
    parser.add_argument("--workspace", type=Path, default=Path(".repomind"),
                        help="Index and retained snapshots (default: .repomind)")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List indexed repositories")
    importing = commands.add_parser("import", help="Index a public GitHub URL or local Git repository")
    importing.add_argument("source")
    searching = commands.add_parser("search", help="Search a repository snapshot")
    searching.add_argument("repository_id")
    searching.add_argument("name", nargs="?", default="")
    searching.add_argument("--file", dest="file_path")
    searching.add_argument("--mode", choices=("symbol", "bm25", "dense", "hybrid"), default="hybrid")
    searching.add_argument("--snapshot", help="Full commit SHA or snapshot ID")
    searching.add_argument("--limit", type=int, default=5)
    indexing = commands.add_parser("index-embeddings", help="Build vectors for one snapshot")
    indexing.add_argument("repository_id")
    indexing.add_argument("--snapshot", help="Full commit SHA or snapshot ID")
    indexing.add_argument("--model", default="text-embedding-v4")
    indexing.add_argument("--dimension", type=int)
    showing = commands.add_parser("show", help="Read one symbol from its retained Git snapshot")
    showing.add_argument("symbol_id")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s %(message)s")
    service = RepoMindService(args.workspace)
    try:
        if args.command == "import":
            result = service.import_source(args.source)
            print(f"repository={result.repository.id} name={result.repository.name}")
            print(f"snapshot={result.snapshot.id} commit={result.snapshot.commit_sha}")
            print(f"files={result.file_count} symbols={result.symbol_count} parse_errors={result.parse_error_count}")
            if result.duplicate:
                print("Already indexed; no duplicate rows created")
            if result.warning:
                print(f"warning: {result.warning}")
        elif args.command == "list":
            for repo, active_id in service.index.list_repositories():
                snapshot = service.index.get_snapshot(active_id) if active_id else None
                print(f"{repo.id}  {repo.name}  {snapshot.commit_sha if snapshot else '-'}  {repo.source}")
        elif args.command in {"search", "index-embeddings"}:
            provider = None
            if args.command == "index-embeddings" or (args.mode in {"dense", "hybrid"}
                                                       and os.environ.get("DASHSCOPE_API_KEY")):
                provider = DashScopeEmbedding(model=getattr(args, "model", "text-embedding-v4"),
                    dimension=getattr(args, "dimension", None),
                    base_url=os.environ.get("DASHSCOPE_BASE_URL",
                                            "https://dashscope.aliyuncs.com/compatible-mode/v1"))
            retrieval = HybridSearch(service, provider)
            if args.command == "index-embeddings":
                done, failed = retrieval.index_embeddings(args.repository_id, snapshot=args.snapshot)
                print(f"embedded={done} failed={failed} model={provider.model_id} dimension={provider.dimension}")
            elif not args.name:
                # The phase-one `search REPO --file PATH` listing remains available.
                selected = args.snapshot
                if selected:
                    resolved = (service.index.get_snapshot(selected)
                                or service.index.find_snapshot_by_commit(args.repository_id, selected))
                    if resolved is None or resolved.repository_id != args.repository_id:
                        raise KeyError("Repository or snapshot was not indexed")
                    selected = resolved.id
                for symbol in service.index.find_symbols(args.repository_id, file_path=args.file_path,
                                                         snapshot_id=selected):
                    print(f"{symbol.id}  {symbol.file_path}:{symbol.start_line}-{symbol.end_line}  "
                          f"{symbol.kind} {symbol.qualified_name}")
            else:
                hits = retrieval.search(args.repository_id, args.name, mode=args.mode,
                                        snapshot=args.snapshot, limit=args.limit)
                if args.file_path:
                    hits = tuple(hit for hit in hits if hit.file_path == args.file_path)
                for rank, hit in enumerate(hits, 1):
                    print(f"{rank}. {hit.symbol_name}  {hit.file_path}:{hit.start_line}-{hit.end_line} "
                          f"score={hit.score:.6g} via={hit.retrieval_source} "
                          f"commit={hit.commit_sha} chunk={hit.chunk_id}")
        elif args.command == "show":
            symbol, source = service.read_symbol(args.symbol_id)
            snapshot = service.index.get_snapshot(symbol.snapshot_id)
            print(f"{symbol.file_path}:{symbol.start_line}-{symbol.end_line} commit={snapshot.commit_sha}")
            for line_number, line in enumerate(source.splitlines(), symbol.start_line):
                print(f"{line_number:>5} {line}")
    except (SourceError, GitImportError, KeyError, RuntimeError, ValueError, EmbeddingError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0
