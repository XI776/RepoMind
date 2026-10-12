"""Replaceable embedding provider and bounded DashScope HTTP adapter."""

import json
import os
import time
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EmbeddingError(RuntimeError):
    pass


class EmbeddingProvider(Protocol):
    model_id: str
    dimension: int | None
    max_batch: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class DashScopeEmbedding:
    def __init__(self, model: str = "text-embedding-v4", *, dimension: int | None = None,
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 timeout: float = 20, max_retries: int = 2, max_batch: int = 10) -> None:
        if not 1 <= max_batch <= 25 or timeout <= 0 or max_retries < 0:
            raise ValueError("Invalid embedding request limits")
        key = os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise EmbeddingError("DASHSCOPE_API_KEY is missing")
        if not base_url.startswith("https://"):
            raise ValueError("Embedding endpoint must use HTTPS")
        self._key = key
        self._url = base_url.rstrip("/") + "/embeddings"
        self.model_id = f"{model}:dim={dimension if dimension is not None else 'default'}"
        self._model = model
        self.dimension = dimension
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_batch = max_batch

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if len(texts) > self.max_batch:
            raise ValueError("Embedding batch exceeds configured limit")
        body: dict[str, object] = {"model": self._model, "input": texts, "encoding_format": "float"}
        if self.dimension is not None:
            body["dimensions"] = self.dimension
        request = Request(self._url, data=json.dumps(body).encode("utf-8"), method="POST",
                          headers={"Authorization": "Bearer " + self._key,
                                   "Content-Type": "application/json"})
        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                data = sorted(payload["data"], key=lambda item: item["index"])
                if len(data) != len(texts) or [item["index"] for item in data] != list(range(len(texts))):
                    raise EmbeddingError("Embedding response count or order is invalid")
                vectors = [[float(value) for value in item["embedding"]] for item in data]
                dimensions = {len(vector) for vector in vectors}
                if len(dimensions) != 1 or 0 in dimensions or (
                    self.dimension is not None and dimensions != {self.dimension}
                ):
                    raise EmbeddingError("Embedding response dimension is invalid")
                self.dimension = dimensions.pop()
                return vectors
            except HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == self.max_retries:
                    raise EmbeddingError(f"Embedding HTTP status {exc.code}") from exc
            except (URLError, TimeoutError) as exc:
                if attempt == self.max_retries:
                    raise EmbeddingError("Embedding request failed or timed out") from exc
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise EmbeddingError("Embedding response is malformed") from exc
            time.sleep(min(2 ** attempt, 4))
        raise EmbeddingError("Embedding retry limit reached")

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
