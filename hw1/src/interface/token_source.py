from __future__ import annotations

from typing import Protocol


class TokenSource(Protocol):
    def read_token(self) -> str | None: ...
