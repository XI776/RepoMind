"""SQLite-backed exact cosine scan keyed by chunk ID and model ID."""

from array import array
from hashlib import sha256
import math
import sys
from typing import Sequence

from repomind.storage import SQLiteIndex


def _unit(values: Sequence[float]) -> tuple[float, ...]:
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Vector must contain finite values")
    norm = math.sqrt(sum(float(value) ** 2 for value in values))
    if norm == 0:
        raise ValueError("Zero vector cannot be indexed or queried")
    return tuple(float(value) / norm for value in values)


class SQLiteVectorStore:
    def __init__(self, index: SQLiteIndex) -> None:
        self.index = index

    def has_current(self, chunk_id: str, model_id: str, input_hash: str) -> bool:
        row = self.index.vector(chunk_id, model_id)
        return row is not None and row["input_hash"] == input_hash

    def put(self, chunk_id: str, model_id: str, repository_id: str, snapshot_id: str,
            input_text: str, values: Sequence[float]) -> None:
        row = self.index.chunk_row(chunk_id)
        if row is None or row["repository_id"] != repository_id or row["snapshot_id"] != snapshot_id:
            raise ValueError("Vector chunk does not belong to the requested snapshot")
        vector = array("d", _unit(values))
        existing_dimension = self.index.vector_dimension(repository_id, snapshot_id, model_id)
        if existing_dimension is not None and existing_dimension != len(vector):
            raise ValueError("Embedding dimensions differ within one model index")
        if sys.byteorder != "little":
            vector.byteswap()
        self.index.save_vector(chunk_id, model_id, repository_id, snapshot_id,
                               sha256(input_text.encode()).hexdigest(), len(vector), vector.tobytes())

    def search(self, repository_id: str, snapshot_id: str, model_id: str,
               query: Sequence[float], limit: int) -> tuple[tuple[str, float], ...]:
        needle = _unit(query)
        scored: list[tuple[str, float]] = []
        for row in self.index.vector_rows(repository_id, snapshot_id, model_id):
            if row["dimension"] != len(needle):
                raise ValueError("Query and indexed vector dimensions differ")
            hay = array("d")
            hay.frombytes(row["vector"])
            if sys.byteorder != "little":
                hay.byteswap()
            if len(hay) != len(needle):
                raise ValueError("Stored vector bytes do not match metadata")
            scored.append((row["chunk_id"], sum(a * b for a, b in zip(needle, hay))))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return tuple(scored[:limit])
