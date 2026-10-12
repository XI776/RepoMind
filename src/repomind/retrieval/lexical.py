"""Turn code identifiers into FTS5 terms; never pass raw user syntax to MATCH."""

import re

_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_PARTS = re.compile(r"[A-Z]+(?=[A-Z][a-z]|$)|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def terms(text: str, *, unique: bool = True) -> tuple[str, ...]:
    found: list[str] = []
    for word in _WORDS.findall(text):
        found.append(word.casefold())
        pieces = _PARTS.findall(word)
        if len(pieces) > 1:
            found.extend(piece.casefold() for piece in pieces)
    return tuple(dict.fromkeys(found)) if unique else tuple(found)


def index_terms(text: str) -> str:
    # FTS5 unicode61 sees spaces as separators. The original identifier and its
    # components are both indexed, including camelCase and acronym boundaries.
    return " ".join(terms(text, unique=False))


def match_query(text: str) -> str:
    # Quoted terms make operators, quotes, dots and parentheses harmless.
    return " OR ".join('"' + word.replace('"', '""') + '"' for word in terms(text)[:32])
