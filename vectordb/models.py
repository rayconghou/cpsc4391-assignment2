from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryResult:
    """One ranked hit from a vector similarity query."""

    id: int
    text: str
    score: float
