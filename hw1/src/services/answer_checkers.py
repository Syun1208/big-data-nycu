from __future__ import annotations

import math

from src.data.classes.records import AnswerCheck
from src.utils.answer_normalization import (
    NUMERIC_ABSOLUTE_TOLERANCE,
    extract_robust_answer_text,
    extract_robust_gsm8k_number,
    normalize_gsm8k_gold,
    robust_math_is_equiv,
)


class Gsm8kAnswerChecker:
    @property
    def evaluation_protocol(self) -> str:
        return "robust_generation_numeric_match"

    def check(
        self,
        *,
        completion: str,
        gold_raw: str,
    ) -> AnswerCheck:
        gold = normalize_gsm8k_gold(value=gold_raw)
        prediction, extraction_source = extract_robust_gsm8k_number(completion=completion)
        correct = prediction is not None and math.isclose(
            float(prediction),
            float(gold),
            rel_tol=0.0,
            abs_tol=NUMERIC_ABSOLUTE_TOLERANCE,
        )
        return AnswerCheck(
            pred_answer=None if prediction is None else str(prediction),
            gold_answer=str(gold),
            correct=correct,
            parsed=prediction is not None,
            extraction_source=extraction_source,
        )


class MathAnswerChecker:
    @property
    def evaluation_protocol(self) -> str:
        return "robust_generation_math_equivalence"

    def check(
        self,
        *,
        completion: str,
        gold_raw: str,
    ) -> AnswerCheck:
        prediction, extraction_source = extract_robust_answer_text(completion=completion)
        correct = prediction is not None and robust_math_is_equiv(
            prediction=prediction,
            gold=gold_raw,
        )
        return AnswerCheck(
            pred_answer=None if prediction is None else str(prediction),
            gold_answer=str(gold_raw),
            correct=correct,
            parsed=prediction is not None,
            extraction_source=extraction_source,
        )
