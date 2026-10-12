"""Snapshot-scoped retrieval; every result is revalidated against its Git blob."""

from hashlib import sha256
import logging

from repomind.models import SearchHit
from repomind.retrieval.embedding import EmbeddingError, EmbeddingProvider
from repomind.retrieval.fusion import rrf
from repomind.retrieval.vector import SQLiteVectorStore
from repomind.service import RepoMindService

log = logging.getLogger(__name__)
MAX_EMBED_CHARS = 3000


class HybridSearch:
    def __init__(self, service: RepoMindService,
                 provider: EmbeddingProvider | None = None) -> None:
        self.service = service
        self.provider = provider
        self.vectors = SQLiteVectorStore(service.index)

    def _snapshot(self, repository_id: str, selector: str | None) -> str:
        index = self.service.index
        snapshot = index.get_snapshot(selector) if selector else index.get_active_snapshot(repository_id)
        if snapshot is None and selector:
            snapshot = index.find_snapshot_by_commit(repository_id, selector)
        if snapshot is None or snapshot.repository_id != repository_id:
            raise KeyError("Repository or snapshot was not indexed")
        return snapshot.id

    def _hit(self, chunk_id: str, score: float, source: str) -> SearchHit:
        row, content = self.service.read_chunk(chunk_id)
        return SearchHit(chunk_id=chunk_id, repository_id=str(row["repository_id"]),
                         snapshot_id=str(row["snapshot_id"]), commit_sha=str(row["commit_sha"]),
                         symbol_id=str(row["symbol_id"]), file_path=str(row["file_path"]),
                         symbol_name=str(row["qualified_name"]), start_line=int(row["start_line"]),
                         end_line=int(row["end_line"]), content=content, score=score,
                         retrieval_source=source, channel_ranks={}, channel_scores={source: score})

    @staticmethod
    def _symbol_score(query: str, name: str, qualified: str) -> float:
        q = query.casefold().strip()
        if not q:
            return 0
        n, full = name.casefold(), qualified.casefold()
        if q == full:
            return 6
        if "." in q and full.endswith("." + q):
            return 5
        if q == n:
            return 4
        if n.startswith(q):
            return 3
        if q in full:
            return 2
        return 0

    def symbol(self, repository_id: str, query: str, *, snapshot: str | None = None,
               limit: int = 5) -> tuple[SearchHit, ...]:
        snapshot_id = self._snapshot(repository_id, snapshot)
        candidates = []
        for row in self.service.index.symbol_candidates(snapshot_id):
            score = self._symbol_score(query, str(row["name"]), str(row["qualified_name"]))
            if score:
                candidates.append((str(row["chunk_id"]), score, str(row["path"]), int(row["start_line"])))
        candidates.sort(key=lambda item: (-item[1], item[2], item[3], item[0]))
        return tuple(self._hit(chunk_id, score, "symbol") for chunk_id, score, _, _ in candidates[:limit])

    def bm25(self, repository_id: str, query: str, *, snapshot: str | None = None,
             limit: int = 5) -> tuple[SearchHit, ...]:
        snapshot_id = self._snapshot(repository_id, snapshot)
        self.service.ensure_search_index(snapshot_id)
        return tuple(self._hit(chunk_id, score, "bm25") for chunk_id, score in
                     self.service.index.bm25_candidates(snapshot_id, query, limit))

    def index_embeddings(self, repository_id: str, *, snapshot: str | None = None) -> tuple[int, int]:
        if self.provider is None:
            raise EmbeddingError("Embedding provider is unavailable")
        snapshot_id = self._snapshot(repository_id, snapshot)
        self.service.ensure_search_index(snapshot_id)
        pending: list[tuple[str, str]] = []
        for chunk_id in self.service.index.indexed_chunks(snapshot_id):
            row, content = self.service.read_chunk(chunk_id)
            text = str(row["qualified_name"]) + "\n" + content[:MAX_EMBED_CHARS]
            digest = sha256(text.encode()).hexdigest()
            if not self.vectors.has_current(chunk_id, self.provider.model_id, digest):
                pending.append((chunk_id, text))
        done = failed = 0
        batch_size = min(self.provider.max_batch, 10)
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            try:
                vectors = self.provider.embed_documents([text for _, text in batch])
                if len(vectors) != len(batch):
                    raise EmbeddingError("Provider returned incorrect batch length")
                for (chunk_id, text), vector in zip(batch, vectors):
                    try:
                        self.vectors.put(chunk_id, self.provider.model_id, repository_id,
                                         snapshot_id, text, vector)
                        done += 1
                    except ValueError as exc:
                        failed += 1
                        log.warning("embedding_rejected chunk=%s reason=%s", chunk_id, exc)
            except EmbeddingError as exc:
                log.warning("embedding_batch_failed count=%d reason=%s", len(batch), exc)
                for chunk_id, text in batch:
                    try:
                        vector = self.provider.embed_documents([text])[0]
                        self.vectors.put(chunk_id, self.provider.model_id, repository_id,
                                         snapshot_id, text, vector)
                        done += 1
                    except (EmbeddingError, ValueError, IndexError) as item_exc:
                        failed += 1
                        log.warning("embedding_chunk_failed chunk=%s reason=%s", chunk_id, item_exc)
        return done, failed

    def dense(self, repository_id: str, query: str, *, snapshot: str | None = None,
              limit: int = 5) -> tuple[SearchHit, ...]:
        if self.provider is None:
            raise EmbeddingError("Embedding provider is unavailable")
        snapshot_id = self._snapshot(repository_id, snapshot)
        query_vector = self.provider.embed_query(query)
        return tuple(self._hit(chunk_id, score, "dense") for chunk_id, score in
                     self.vectors.search(repository_id, snapshot_id,
                                         self.provider.model_id, query_vector, limit))

    def search(self, repository_id: str, query: str, *, mode: str = "hybrid",
               snapshot: str | None = None, limit: int = 5, k: int = 60) -> tuple[SearchHit, ...]:
        if mode not in {"symbol", "bm25", "dense", "hybrid"} or limit < 1:
            raise ValueError("Invalid search mode or limit")
        if mode != "hybrid":
            return getattr(self, mode)(repository_id, query, snapshot=snapshot, limit=limit)
        budget = max(limit, 20)
        channels = {
            "symbol": self.symbol(repository_id, query, snapshot=snapshot, limit=budget),
            "bm25": self.bm25(repository_id, query, snapshot=snapshot, limit=budget),
        }
        if self.provider is not None:
            try:
                channels["dense"] = self.dense(repository_id, query, snapshot=snapshot, limit=budget)
            except (EmbeddingError, ValueError) as exc:
                log.warning("dense_search_failed reason=%s", exc)
        return rrf(channels, limit=limit, k=k)
