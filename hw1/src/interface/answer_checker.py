from __future__ import annotations

from typing import Protocol

from src.data.classes.records import AnswerCheck


class AnswerChecker(Protocol):
    @property
    def evaluation_protocol(self) -> str: ...

    def check(
        self,
        *,
        completion: str,
        gold_raw: str,
    ) -> AnswerCheck: ...
