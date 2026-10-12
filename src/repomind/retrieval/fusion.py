"""Reciprocal rank fusion over stable chunk identities."""

from dataclasses import replace

from repomind.models import SearchHit


def rrf(channels: dict[str, tuple[SearchHit, ...]], limit: int = 5,
        k: int = 60) -> tuple[SearchHit, ...]:
    if k <= 0 or limit < 0:
        raise ValueError("RRF k must be positive and limit nonnegative")
    merged: dict[str, SearchHit] = {}
    ranks: dict[str, dict[str, int]] = {}
    scores: dict[str, dict[str, float]] = {}
    for channel, hits in channels.items():
        seen: set[str] = set()
        for hit in hits:
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            rank = len(seen)
            merged.setdefault(hit.chunk_id, hit)
            ranks.setdefault(hit.chunk_id, {})[channel] = rank
            scores.setdefault(hit.chunk_id, {})[channel] = hit.score
    result = [replace(hit, score=sum(1 / (k + rank) for rank in ranks[chunk_id].values()),
                      retrieval_source="+".join(sorted(ranks[chunk_id])),
                      channel_ranks=ranks[chunk_id], channel_scores=scores[chunk_id])
              for chunk_id, hit in merged.items()]
    result.sort(key=lambda hit: (-hit.score, hit.chunk_id))
    return tuple(result[:limit])
