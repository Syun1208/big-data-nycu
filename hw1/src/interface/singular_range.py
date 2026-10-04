from __future__ import annotations

from typing import Protocol


class SingularRangeSelector(Protocol):
    def select_start(
        self,
        *,
        singular_count: int,
        rank: int,
    ) -> int: ...
