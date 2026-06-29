from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PrefixCache:
    """Hash-based KV block reuse for shared prompt prefixes."""

    block_size: int
    entries: dict[str, list[int]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def lookup(self, prefix_hash: str) -> list[int] | None:
        blocks = self.entries.get(prefix_hash)
        if blocks:
            self.hits += 1
            return list(blocks)
        self.misses += 1
        return None

    def register(self, prefix_hash: str, block_ids: list[int]) -> None:
        self.entries[prefix_hash] = list(block_ids)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict:
        return {
            "entries": len(self.entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(self.hit_rate * 100, 1),
        }
