from __future__ import annotations


class TopSingularRange:
    def select_start(
        self,
        *,
        singular_count: int,
        rank: int,
    ) -> int:
        return 0


class PositionSingularRange:
    def __init__(
        self,
        *,
        numerator: int,
        denominator: int,
    ) -> None:
        if not 0 <= numerator < denominator:
            raise ValueError(f"position must be in [0, 1): {numerator}/{denominator}")
        self._numerator = numerator
        self._denominator = denominator

    def select_start(
        self,
        *,
        singular_count: int,
        rank: int,
    ) -> int:
        start = singular_count * self._numerator // self._denominator
        if start + rank > singular_count:
            raise ValueError(f"range [{start}, {start + rank}) exceeds {singular_count} singular values")
        return start


class BottomSingularRange:
    def select_start(
        self,
        *,
        singular_count: int,
        rank: int,
    ) -> int:
        return singular_count - rank
