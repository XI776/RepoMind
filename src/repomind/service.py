"""Stage-one orchestration; future retrieval can consume the same index."""

from dataclasses import dataclass
from hashlib import sha256
import logging
from pathlib import Path

from repomind.ingestion import GitImporter
from repomind.ingestion.git import GitFile
from repomind.models import Repository, RepositorySnapshot, StoredSymbol
from repomind.parsing import decode_python, parse_python
from repomind.storage import SQLiteIndex

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ImportResult:
    repository: Repository
    snapshot: RepositorySnapshot
    file_count: int
    symbol_count: int
    parse_error_count: int
    duplicate: bool
    warning: str | None


class RepoMindService:
    def __init__(self, workspace: Path) -> None:
        workspace = workspace.expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        self.importer = GitImporter(workspace)
        self.index = SQLiteIndex(workspace / "index.sqlite3")

    def import_source(self, source: str) -> ImportResult:
        imported = self.importer.import_source(source)
        if self.index.has_snapshot(imported.snapshot.id):
            stored_snapshot = self.index.get_snapshot(imported.snapshot.id)
            files = self.index.list_files(imported.snapshot.id)
            symbols = self.index.find_symbols(imported.repository.id, snapshot_id=imported.snapshot.id)
            return ImportResult(imported.repository, stored_snapshot, len(files), len(symbols),
                                sum(f.parse_status != "parsed" for f in files), True, imported.warning)
        parsed = tuple(parse_python(imported.snapshot.id, entry.path, entry.blob_oid,
                                    self.importer.read_blob(imported.snapshot, entry))
                       for entry in imported.files)
        self.index.save_snapshot(imported.repository, imported.snapshot, parsed)
        error_count = sum(item.file.parse_status != "parsed" for item in parsed)
        log.info("index_published repo=%s revision=%s files=%d symbols=%d errors=%d",
                 imported.repository.id, imported.snapshot.id, len(parsed),
                 sum(len(item.symbols) for item in parsed), error_count)
        return ImportResult(imported.repository, imported.snapshot, len(parsed),
                            sum(len(item.symbols) for item in parsed), error_count,
                            False, imported.warning)

    def read_symbol(self, symbol_id: str) -> tuple[StoredSymbol, str]:
        symbol = self.index.get_symbol(symbol_id)
        if symbol is None:
            raise KeyError(f"Unknown symbol: {symbol_id}")
        snapshot = self.index.get_snapshot(symbol.snapshot_id)
        if snapshot is None:
            raise RuntimeError("Indexed snapshot is missing")
        file = next((f for f in self.index.list_files(snapshot.id) if f.path == symbol.file_path), None)
        if file is None:
            raise RuntimeError("Indexed source file is missing")
        raw = self.importer.read_blob(snapshot, GitFile(file.path, file.blob_oid, file.size_bytes))
        if sha256(raw).hexdigest() != file.content_hash:
            raise RuntimeError("Snapshot content does not match the indexed file")
        lines = decode_python(raw).splitlines(keepends=True)
        if not (1 <= symbol.start_line <= symbol.end_line <= len(lines)):
            raise RuntimeError("Indexed line range is invalid")
        return symbol, "".join(lines[symbol.start_line - 1:symbol.end_line])

    def read_chunk(self, chunk_id: str) -> tuple[dict[str, object], str]:
        row = self.index.chunk_row(chunk_id)
        if row is None:
            raise KeyError(f"Unknown chunk: {chunk_id}")
        symbol, source = self.read_symbol(str(row["symbol_id"]))
        if (symbol.start_line != row["start_line"] or symbol.end_line != row["end_line"]
                or sha256(source.encode("utf-8")).hexdigest() != row["content_hash"]):
            raise RuntimeError("Chunk range or content hash differs from retained snapshot")
        return row, source

    def ensure_search_index(self, snapshot_id: str) -> None:
        """Lazily migrate phase-one snapshots into the rebuildable FTS cache."""
        for chunk_id, _symbol_id, qualified_name in self.index.missing_search_docs(snapshot_id):
            _, content = self.read_chunk(chunk_id)
            self.index.add_search_doc(chunk_id, snapshot_id, qualified_name, content)
