"""Data records; no database or Git behavior lives here."""

from dataclasses import dataclass
from dataclasses import field
from hashlib import sha256


def stable_id(*parts: str) -> str:
    """Hash length-delimited parts so names cannot collide by concatenation."""
    digest = sha256()
    for part in parts:
        raw = part.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Repository:
    id: str
    name: str
    source_type: str  # github | local
    source: str


@dataclass(frozen=True, slots=True)
class RepositorySnapshot:
    id: str
    repository_id: str
    commit_sha: str
    index_config_id: str
    git_dir: str  # Retained bare Git repository containing the indexed commit.
    imported_at: str


@dataclass(frozen=True, slots=True)
class SourceFile:
    id: str
    snapshot_id: str
    path: str  # POSIX path relative to the repository root.
    blob_oid: str
    content_hash: str
    size_bytes: int
    parse_status: str  # parsed | syntax_error | encoding_error | skipped
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CodeSymbol:
    id: str
    file_id: str
    name: str
    qualified_name: str
    kind: str  # class | function | async_function | method | async_method
    signature: str
    docstring: str | None
    start_line: int  # Includes decorators, if any.
    definition_line: int
    end_line: int
    source: str  # Parsed source, not persisted as authoritative evidence.


@dataclass(frozen=True, slots=True)
class CodeChunk:
    id: str
    symbol_id: str
    start_line: int
    end_line: int
    content_hash: str


@dataclass(frozen=True, slots=True)
class ImportRelation:
    id: str
    file_id: str
    scope: str
    kind: str  # import | from
    module: str | None
    names: tuple[str, ...]
    level: int
    line: int


@dataclass(frozen=True, slots=True)
class ParsedFile:
    file: SourceFile
    symbols: tuple[CodeSymbol, ...]
    chunks: tuple[CodeChunk, ...]
    imports: tuple[ImportRelation, ...]


@dataclass(frozen=True, slots=True)
class StoredSymbol:
    """Queryable metadata; source is read from the retained Git blob."""

    id: str
    repository_id: str
    snapshot_id: str
    file_path: str
    blob_oid: str
    name: str
    qualified_name: str
    kind: str
    signature: str
    docstring: str | None
    start_line: int
    definition_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: str
    repository_id: str
    snapshot_id: str
    commit_sha: str
    symbol_id: str
    file_path: str
    symbol_name: str
    start_line: int
    end_line: int
    content: str
    score: float
    retrieval_source: str
    channel_ranks: dict[str, int] = field(default_factory=dict)
    channel_scores: dict[str, float] = field(default_factory=dict)
