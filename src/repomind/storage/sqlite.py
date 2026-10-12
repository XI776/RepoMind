"""Transactional structural index. Git blobs remain authoritative source text."""

from contextlib import closing, contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Iterator

from repomind.models import ImportRelation, ParsedFile, Repository, RepositorySnapshot, SourceFile, StoredSymbol
from repomind.retrieval.lexical import index_terms, match_query


class SQLiteIndex:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            try:
                yield connection
                connection.commit()
            except sqlite3.Error:
                connection.rollback()
                raise

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS repositories (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, source_type TEXT NOT NULL,
                    source TEXT NOT NULL UNIQUE, active_snapshot_id TEXT
                );
                CREATE TABLE IF NOT EXISTS snapshots (
                    id TEXT PRIMARY KEY, repository_id TEXT NOT NULL REFERENCES repositories(id),
                    commit_sha TEXT NOT NULL, index_config_id TEXT NOT NULL,
                    git_dir TEXT NOT NULL, imported_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('active', 'ready')),
                    UNIQUE(repository_id, commit_sha, index_config_id)
                );
                CREATE TABLE IF NOT EXISTS files (
                    id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
                    path TEXT NOT NULL, blob_oid TEXT NOT NULL, content_hash TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL, parse_status TEXT NOT NULL, error TEXT,
                    UNIQUE(snapshot_id, path)
                );
                CREATE TABLE IF NOT EXISTS symbols (
                    id TEXT PRIMARY KEY, file_id TEXT NOT NULL REFERENCES files(id),
                    name TEXT NOT NULL, qualified_name TEXT NOT NULL, kind TEXT NOT NULL,
                    signature TEXT NOT NULL, docstring TEXT, start_line INTEGER NOT NULL,
                    definition_line INTEGER NOT NULL, end_line INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS symbols_name_idx ON symbols(name);
                CREATE INDEX IF NOT EXISTS symbols_file_idx ON symbols(file_id);
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, symbol_id TEXT NOT NULL REFERENCES symbols(id),
                    start_line INTEGER NOT NULL, end_line INTEGER NOT NULL, content_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS imports (
                    id TEXT PRIMARY KEY, file_id TEXT NOT NULL REFERENCES files(id),
                    scope TEXT NOT NULL, kind TEXT NOT NULL, module TEXT,
                    names_json TEXT NOT NULL, level INTEGER NOT NULL, line INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS imports_file_idx ON imports(file_id);
                CREATE TABLE IF NOT EXISTS chunk_search_docs (
                    id INTEGER PRIMARY KEY, chunk_id TEXT NOT NULL UNIQUE REFERENCES chunks(id),
                    snapshot_id TEXT NOT NULL REFERENCES snapshots(id)
                );
                CREATE INDEX IF NOT EXISTS chunk_docs_snapshot_idx ON chunk_search_docs(snapshot_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                    name_terms, content, tokenize='unicode61'
                );
                CREATE TABLE IF NOT EXISTS chunk_vectors (
                    chunk_id TEXT NOT NULL REFERENCES chunks(id), model_id TEXT NOT NULL,
                    repository_id TEXT NOT NULL, snapshot_id TEXT NOT NULL,
                    input_hash TEXT NOT NULL, dimension INTEGER NOT NULL, vector BLOB NOT NULL,
                    PRIMARY KEY(chunk_id, model_id)
                );
                CREATE INDEX IF NOT EXISTS vectors_scope_idx
                    ON chunk_vectors(repository_id, snapshot_id, model_id);
            """)

    def has_snapshot(self, snapshot_id: str) -> bool:
        with self._connect() as db:
            return db.execute("SELECT 1 FROM snapshots WHERE id = ?", (snapshot_id,)).fetchone() is not None

    def save_snapshot(self, repository: Repository, snapshot: RepositorySnapshot,
                      parsed: tuple[ParsedFile, ...]) -> bool:
        """Publish one complete revision in a SQLite transaction; False means duplicate."""
        with self._connect() as db:
            if db.execute("SELECT 1 FROM snapshots WHERE id = ?", (snapshot.id,)).fetchone():
                return False
            db.execute("INSERT OR IGNORE INTO repositories(id, name, source_type, source) VALUES (?, ?, ?, ?)",
                       (repository.id, repository.name, repository.source_type, repository.source))
            db.execute("UPDATE snapshots SET status = 'ready' WHERE repository_id = ? AND status = 'active'",
                       (repository.id,))
            db.execute("INSERT INTO snapshots VALUES (?, ?, ?, ?, ?, ?, 'active')",
                       (snapshot.id, repository.id, snapshot.commit_sha, snapshot.index_config_id,
                        snapshot.git_dir, snapshot.imported_at))
            for item in parsed:
                file = item.file
                db.execute("INSERT INTO files VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                           (file.id, file.snapshot_id, file.path, file.blob_oid, file.content_hash,
                            file.size_bytes, file.parse_status, file.error))
                db.executemany("INSERT INTO symbols VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
                    (s.id, s.file_id, s.name, s.qualified_name, s.kind, s.signature,
                     s.docstring, s.start_line, s.definition_line, s.end_line)
                    for s in item.symbols
                ])
                db.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)", [
                    (c.id, c.symbol_id, c.start_line, c.end_line, c.content_hash)
                    for c in item.chunks
                ])
                symbols_by_id = {symbol.id: symbol for symbol in item.symbols}
                for chunk in item.chunks:
                    symbol = symbols_by_id[chunk.symbol_id]
                    self._insert_search_doc(db, chunk.id, snapshot.id,
                                            symbol.qualified_name, symbol.source)
                db.executemany("INSERT INTO imports VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
                    (i.id, i.file_id, i.scope, i.kind, i.module, json.dumps(i.names), i.level, i.line)
                    for i in item.imports
                ])
            db.execute("UPDATE repositories SET active_snapshot_id = ? WHERE id = ?",
                       (snapshot.id, repository.id))
            return True

    @staticmethod
    def _insert_search_doc(db: sqlite3.Connection, chunk_id: str, snapshot_id: str,
                           qualified_name: str, content: str) -> None:
        cursor = db.execute("INSERT OR IGNORE INTO chunk_search_docs(chunk_id, snapshot_id) VALUES (?, ?)",
                            (chunk_id, snapshot_id))
        if cursor.rowcount:
            rowid = cursor.lastrowid
            db.execute("INSERT INTO chunk_fts(rowid, name_terms, content) VALUES (?, ?, ?)",
                       (rowid, index_terms(qualified_name), index_terms(content)))

    def missing_search_docs(self, snapshot_id: str) -> tuple[tuple[str, str, str], ...]:
        """Old phase-one databases can be indexed without reimporting a commit."""
        with self._connect() as db:
            rows = db.execute("""SELECT c.id, c.symbol_id, s.qualified_name FROM chunks c
                JOIN symbols s ON s.id=c.symbol_id JOIN files f ON f.id=s.file_id
                LEFT JOIN chunk_search_docs d ON d.chunk_id=c.id
                WHERE f.snapshot_id=? AND d.id IS NULL""", (snapshot_id,)).fetchall()
        return tuple((r["id"], r["symbol_id"], r["qualified_name"]) for r in rows)

    def add_search_doc(self, chunk_id: str, snapshot_id: str, qualified_name: str,
                       content: str) -> None:
        with self._connect() as db:
            self._insert_search_doc(db, chunk_id, snapshot_id, qualified_name, content)

    def chunk_row(self, chunk_id: str) -> dict[str, object] | None:
        with self._connect() as db:
            row = db.execute("""SELECT c.id AS chunk_id, c.symbol_id, c.start_line,
                c.end_line, c.content_hash, s.qualified_name, f.path AS file_path,
                f.snapshot_id, p.repository_id, p.commit_sha FROM chunks c
                JOIN symbols s ON s.id=c.symbol_id JOIN files f ON f.id=s.file_id
                JOIN snapshots p ON p.id=f.snapshot_id WHERE c.id=?""", (chunk_id,)).fetchone()
        return dict(row) if row else None

    def symbol_candidates(self, snapshot_id: str) -> tuple[dict[str, object], ...]:
        with self._connect() as db:
            rows = db.execute("""SELECT c.id AS chunk_id, s.name, s.qualified_name,
                f.path, c.start_line FROM chunks c JOIN symbols s ON s.id=c.symbol_id
                JOIN files f ON f.id=s.file_id WHERE f.snapshot_id=?""", (snapshot_id,)).fetchall()
        return tuple(dict(r) for r in rows)

    def bm25_candidates(self, snapshot_id: str, query: str, limit: int) -> tuple[tuple[str, float], ...]:
        expression = match_query(query)
        if not expression:
            return ()
        with self._connect() as db:
            rows = db.execute("""SELECT d.chunk_id, bm25(chunk_fts, 4.0, 1.0) AS relevance
                FROM chunk_fts JOIN chunk_search_docs d ON d.id=chunk_fts.rowid
                WHERE chunk_fts MATCH ? AND d.snapshot_id=?
                ORDER BY relevance ASC, d.chunk_id ASC LIMIT ?""",
                              (expression, snapshot_id, limit)).fetchall()
        return tuple((r["chunk_id"], float(r["relevance"])) for r in rows)

    def indexed_chunks(self, snapshot_id: str) -> tuple[str, ...]:
        with self._connect() as db:
            rows = db.execute("SELECT chunk_id FROM chunk_search_docs WHERE snapshot_id=? ORDER BY chunk_id",
                              (snapshot_id,)).fetchall()
        return tuple(r["chunk_id"] for r in rows)

    def vector(self, chunk_id: str, model_id: str) -> dict[str, object] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM chunk_vectors WHERE chunk_id=? AND model_id=?",
                             (chunk_id, model_id)).fetchone()
        return dict(row) if row else None

    def save_vector(self, chunk_id: str, model_id: str, repository_id: str,
                    snapshot_id: str, input_hash: str, dimension: int, vector: bytes) -> None:
        with self._connect() as db:
            db.execute("""INSERT INTO chunk_vectors VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id, model_id) DO UPDATE SET
                input_hash=excluded.input_hash, dimension=excluded.dimension,
                vector=excluded.vector""", (chunk_id, model_id, repository_id,
                                               snapshot_id, input_hash, dimension, vector))

    def vector_rows(self, repository_id: str, snapshot_id: str,
                    model_id: str) -> tuple[dict[str, object], ...]:
        with self._connect() as db:
            rows = db.execute("""SELECT * FROM chunk_vectors WHERE repository_id=?
                AND snapshot_id=? AND model_id=? ORDER BY chunk_id""",
                (repository_id, snapshot_id, model_id)).fetchall()
        return tuple(dict(r) for r in rows)

    def vector_dimension(self, repository_id: str, snapshot_id: str,
                         model_id: str) -> int | None:
        with self._connect() as db:
            row = db.execute("""SELECT dimension FROM chunk_vectors WHERE repository_id=?
                AND snapshot_id=? AND model_id=? LIMIT 1""",
                (repository_id, snapshot_id, model_id)).fetchone()
        return int(row["dimension"]) if row else None

    def list_repositories(self) -> tuple[tuple[Repository, str | None], ...]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM repositories ORDER BY name, id").fetchall()
        return tuple((Repository(r["id"], r["name"], r["source_type"], r["source"]),
                      r["active_snapshot_id"]) for r in rows)

    def get_snapshot(self, snapshot_id: str) -> RepositorySnapshot | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM snapshots WHERE id = ?", (snapshot_id,)).fetchone()
        if row is None:
            return None
        return RepositorySnapshot(row["id"], row["repository_id"], row["commit_sha"],
                                  row["index_config_id"], row["git_dir"], row["imported_at"])

    def get_active_snapshot(self, repository_id: str) -> RepositorySnapshot | None:
        with self._connect() as db:
            row = db.execute("SELECT active_snapshot_id FROM repositories WHERE id = ?",
                             (repository_id,)).fetchone()
        return self.get_snapshot(row["active_snapshot_id"]) if row and row["active_snapshot_id"] else None

    def find_snapshot_by_commit(self, repository_id: str, commit_sha: str) -> RepositorySnapshot | None:
        with self._connect() as db:
            row = db.execute("SELECT id FROM snapshots WHERE repository_id=? AND commit_sha=?",
                             (repository_id, commit_sha)).fetchone()
        return self.get_snapshot(row["id"]) if row else None

    def list_files(self, snapshot_id: str) -> tuple[SourceFile, ...]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM files WHERE snapshot_id = ? ORDER BY path", (snapshot_id,)).fetchall()
        return tuple(SourceFile(r["id"], r["snapshot_id"], r["path"], r["blob_oid"],
                                r["content_hash"], r["size_bytes"], r["parse_status"], r["error"])
                     for r in rows)

    def list_imports(self, file_id: str) -> tuple[ImportRelation, ...]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM imports WHERE file_id = ? ORDER BY line", (file_id,)).fetchall()
        return tuple(ImportRelation(r["id"], r["file_id"], r["scope"], r["kind"], r["module"],
                                    tuple(json.loads(r["names_json"])), r["level"], r["line"])
                     for r in rows)

    def find_symbols(self, repository_id: str, name: str = "", file_path: str | None = None,
                     snapshot_id: str | None = None) -> tuple[StoredSymbol, ...]:
        snapshot = self.get_snapshot(snapshot_id) if snapshot_id else self.get_active_snapshot(repository_id)
        if snapshot is None or snapshot.repository_id != repository_id:
            return ()
        query = """SELECT s.*, f.path AS file_path, f.blob_oid FROM symbols s
                   JOIN files f ON f.id = s.file_id
                   WHERE f.snapshot_id = ?"""
        params: list[str] = [snapshot.id]
        if name:
            query += " AND (s.name = ? OR s.qualified_name = ?)"
            params.extend((name, name))
        if file_path is not None:
            query += " AND f.path = ?"
            params.append(file_path)
        query += " ORDER BY f.path, s.start_line, s.id"
        with self._connect() as db:
            rows = db.execute(query, params).fetchall()
        return tuple(StoredSymbol(r["id"], repository_id, snapshot.id, r["file_path"], r["blob_oid"],
                                  r["name"], r["qualified_name"], r["kind"], r["signature"],
                                  r["docstring"], r["start_line"], r["definition_line"], r["end_line"])
                     for r in rows)

    def get_symbol(self, symbol_id: str) -> StoredSymbol | None:
        with self._connect() as db:
            row = db.execute("""SELECT s.*, f.path AS file_path, f.blob_oid, f.snapshot_id,
                               p.repository_id FROM symbols s JOIN files f ON f.id = s.file_id
                               JOIN snapshots p ON p.id = f.snapshot_id WHERE s.id = ?""",
                             (symbol_id,)).fetchone()
        if row is None:
            return None
        return StoredSymbol(row["id"], row["repository_id"], row["snapshot_id"], row["file_path"],
                            row["blob_oid"], row["name"], row["qualified_name"], row["kind"],
                            row["signature"], row["docstring"], row["start_line"],
                            row["definition_line"], row["end_line"])
