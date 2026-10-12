from pathlib import Path
import sqlite3

import pytest

from conftest import git
from repomind.retrieval.embedding import EmbeddingError
from repomind.retrieval.fusion import rrf
from repomind.retrieval.lexical import terms
from repomind.retrieval.search import HybridSearch
from repomind.service import RepoMindService


class FakeEmbedding:
    model_id = "fake-v1:dim=2"
    dimension = 2
    max_batch = 3

    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += len(texts)
        return [[1.0, 0.0] if "TcpConnection" in text else [0.0, 1.0]
                for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0] if "connection" in text.lower() else [0.0, 1.0]


class BrokenEmbedding(FakeEmbedding):
    def embed_query(self, text: str) -> list[float]:
        raise EmbeddingError("fake outage")


class OneBadEmbedding(FakeEmbedding):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if any("pkg.connection.HTTPServer" in text for text in texts):
            raise EmbeddingError("one invalid item")
        return super().embed_documents(texts)


@pytest.fixture
def search_fixture(git_repo: Path, tmp_path: Path) -> tuple[RepoMindService, str, str, Path]:
    (git_repo / "pkg" / "connection.py").write_text(
        "class TcpConnection:\n"
        "    def handleRead(self):\n        return 'connection read'\n"
        "    def handle_read(self):\n        return 'connection close'\n"
        "class HTTPServer:\n    def close(self):\n        return 'server closed'\n",
        encoding="utf-8")
    (git_repo / "pkg" / "other.py").write_text(
        "class TcpConnection:\n    def handleRead(self):\n        return 'other read'\n",
        encoding="utf-8")
    git(git_repo, "add", ".")
    git(git_repo, "commit", "-qm", "retrieval fixture")
    service = RepoMindService(tmp_path / "workspace")
    imported = service.import_source(str(git_repo))
    return service, imported.repository.id, imported.snapshot.id, git_repo


def test_symbol_name_priority_and_same_names(search_fixture):
    service, repo, snapshot, _ = search_fixture
    search = HybridSearch(service)
    hits = search.symbol(repo, "TcpConnection.handleRead", snapshot=snapshot)
    assert len(hits) == 2
    assert {h.file_path for h in hits} == {"pkg/connection.py", "pkg/other.py"}
    assert all(h.symbol_name.endswith("TcpConnection.handleRead") for h in hits)
    assert search.symbol(repo, "pkg.connection.TcpConnection.handleRead")[0].file_path == "pkg/connection.py"
    assert search.symbol(repo, "HTTPServer")[0].symbol_name.endswith("HTTPServer")
    assert search.symbol(repo, "handle")


def test_bm25_identifiers_special_chars_and_source_ranges(search_fixture):
    service, repo, snapshot, _ = search_fixture
    search = HybridSearch(service)
    assert {"handle", "read"}.issubset(terms("handleRead"))
    assert {"tcp", "connection"}.issubset(terms("TcpConnection"))
    assert {"http", "server"}.issubset(terms("HTTPServer"))
    for query in ("handleRead", "handle_read", "TcpConnection", "HTTPServer", "HTTPServer.close()", '" OR (*'):
        search.bm25(repo, query)
    hits = search.bm25(repo, "connection close")
    assert hits and all(hit.snapshot_id == snapshot for hit in hits)
    for hit in hits:
        row, source = service.read_chunk(hit.chunk_id)
        assert source == hit.content
        assert row["start_line"] == hit.start_line
        assert row["end_line"] == hit.end_line
    assert search.bm25(repo, "HTTPServer")


def test_repository_and_revision_isolation(search_fixture, tmp_path):
    service, repo, old, source = search_fixture
    other = tmp_path / "other"
    other.mkdir()
    git(other, "init", "-q")
    git(other, "config", "user.name", "Test")
    git(other, "config", "user.email", "test@example.invalid")
    (other / "x.py").write_text("def foreign_only():\n    return 1\n")
    git(other, "add", ".")
    git(other, "commit", "-qm", "initial")
    other_repo = service.import_source(str(other)).repository.id
    search = HybridSearch(service)
    assert not search.bm25(repo, "foreign_only")
    assert search.bm25(other_repo, "foreign_only")
    (source / "pkg" / "connection.py").write_text("def new_only():\n    return 3\n")
    git(source, "add", ".")
    git(source, "commit", "-qm", "new revision")
    new = service.import_source(str(source)).snapshot.id
    assert not search.bm25(repo, "new_only", snapshot=old)
    assert search.bm25(repo, "new_only", snapshot=new)
    assert search.bm25(repo, "handleRead", snapshot=old)
    assert all(h.file_path != "pkg/connection.py" for h in search.bm25(repo, "handleRead", snapshot=new))
    assert all(hit.snapshot_id == old for hit in search.symbol(repo, "handleRead", snapshot=old))


def test_dense_persistence_idempotence_dimension_and_degradation(search_fixture):
    service, repo, snapshot, _ = search_fixture
    fake = FakeEmbedding()
    search = HybridSearch(service, fake)
    done, failed = search.index_embeddings(repo)
    assert done > 0 and failed == 0
    assert search.index_embeddings(repo) == (0, 0)
    assert fake.calls == done
    hits = search.dense(repo, "connection", snapshot=snapshot)
    assert hits and hits[0].score == pytest.approx(1.0)
    assert "connection" in hits[0].content.lower() or "Connection" in hits[0].symbol_name
    reloaded = HybridSearch(RepoMindService(service.importer.workspace), FakeEmbedding())
    assert reloaded.dense(repo, "connection")[0].chunk_id == hits[0].chunk_id
    with pytest.raises(ValueError, match="dimensions"):
        search.vectors.search(repo, snapshot, fake.model_id, [1.0, 0.0, 0.0], 5)
    with pytest.raises(ValueError, match="Zero vector"):
        search.vectors.search(repo, snapshot, fake.model_id, [0.0, 0.0], 5)
    degraded = HybridSearch(service, BrokenEmbedding()).search(repo, "connection")
    assert degraded and all("dense" not in h.channel_ranks for h in degraded)
    assert HybridSearch(service).search(repo, "connection")


def test_rrf_rank_dedup_and_empty(search_fixture):
    service, repo, _, _ = search_fixture
    base = HybridSearch(service).symbol(repo, "handleRead", limit=2)
    one, two = base
    merged = rrf({"symbol": (one, one, two), "bm25": (one, two)}, limit=2)
    assert len(merged) == 2 and merged[0].chunk_id == one.chunk_id
    assert merged[0].channel_ranks == {"symbol": 1, "bm25": 1}
    assert merged[0].score == pytest.approx(2/61)
    assert rrf({"dense": ()}) == ()


def test_old_database_backfill_and_chunk_hash_validation(search_fixture):
    service, repo, snapshot, _ = search_fixture
    with sqlite3.connect(service.index.path) as db:
        db.execute("DELETE FROM chunk_fts")
        db.execute("DELETE FROM chunk_search_docs")
    search = HybridSearch(service)
    assert search.bm25(repo, "handleRead", snapshot=snapshot)
    first_count = len(service.index.indexed_chunks(snapshot))
    search.bm25(repo, "handleRead", snapshot=snapshot)
    assert len(service.index.indexed_chunks(snapshot)) == first_count
    chunk_id = service.index.indexed_chunks(snapshot)[0]
    with sqlite3.connect(service.index.path) as db:
        db.execute("UPDATE chunks SET content_hash='wrong' WHERE id=?", (chunk_id,))
    with pytest.raises(RuntimeError, match="Chunk range or content hash"):
        service.read_chunk(chunk_id)


def test_embedding_failure_keeps_success_and_repeat_import_is_idempotent(search_fixture):
    service, repo, snapshot, source = search_fixture
    provider = OneBadEmbedding()
    search = HybridSearch(service, provider)
    done, failed = search.index_embeddings(repo)
    assert done > 0 and failed >= 1
    assert len(service.index.vector_rows(repo, snapshot, provider.model_id)) == done
    before = len(service.index.indexed_chunks(snapshot))
    assert service.import_source(str(source)).duplicate
    assert len(service.index.indexed_chunks(snapshot)) == before
    assert search.index_embeddings(repo) == (0, failed)
