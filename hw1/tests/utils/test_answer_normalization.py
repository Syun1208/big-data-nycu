from __future__ import annotations

import random
from typing import Any

import pytest

from src.services.answer_checkers import Gsm8kAnswerChecker, MathAnswerChecker
from src.utils import answer_normalization as refactored

RANDOM_CASES = 3000
RANDOM_SEED = 1234
FRAGMENTS = (
    "The answer is: ",
    "The answer is ",
    "Answer: ",
    "Final answer = ",
    "####",
    "\\boxed{",
    "}",
    "{",
    "\\frac",
    "\\dfrac",
    "\\sqrt",
    "\\text{ cm}",
    "\\left(",
    "\\right)",
    "^\\circ",
    "\\%",
    "\\\\",
    "\\\\\\%",
    "$",
    "\\(",
    "\\]",
    ".",
    ",",
    ";",
    "=",
    "/",
    " ",
    "\n",
    ".\n",
    "-",
    "+",
    "0",
    "1",
    "2",
    "12",
    "3.5",
    "1,234",
    "1/2",
    "0.5",
    ".5",
    "x",
    "pi",
)
HANDWRITTEN_COMPLETIONS = (
    "So 3 + 4 = 7.\nThe answer is: 7",
    "The answer is: \\boxed{\\frac{1}{2}}.",
    "Therefore, the answer is 1,234.",
    "#### 42\nmore text",
    "We get \\boxed{\\sqrt{2}} as result",
    "no answer here 12 and 15",
    "Answer: 0.5",
    "The answer is: \\frac12",
    "The answer is: 10\\%",
    "The answer is: x = 3",
    "The answer is: 5\\text{ cm}",
    "The answer is: 1/0",
    "",
)
HANDWRITTEN_GOLDS = ("7", "\\frac{1}{2}", "1234", "42", "\\sqrt{2}", "15", "0.5", "10", "3", "5", "#### 1,234", "1/2")


def build_random_texts() -> list[str]:
    generator = random.Random(RANDOM_SEED)
    return ["".join(generator.choice(FRAGMENTS) for _ in range(generator.randint(0, 8))) for _ in range(RANDOM_CASES)]


TEXTS = list(HANDWRITTEN_COMPLETIONS) + build_random_texts()


def call_safely(function: Any, *args: Any, **kwargs: Any) -> tuple[str, Any]:
    try:
        return "ok", function(*args, **kwargs)
    except Exception as error:
        return "error", type(error).__name__


@pytest.mark.parametrize(
    ("original_name", "refactored_function", "keyword"),
    [
        ("strip_pissa_math_string", refactored.strip_pissa_math_string, "string"),
        ("extract_last_boxed", refactored.extract_last_boxed, "text"),
        ("clean_robust_answer_text", refactored.clean_robust_answer_text, "text"),
        ("extract_robust_answer_text", refactored.extract_robust_answer_text, "completion"),
        ("extract_robust_gsm8k_number", refactored.extract_robust_gsm8k_number, "completion"),
        ("extract_pissa_gsm8k_number", refactored.extract_pissa_gsm8k_number, "completion"),
        ("simple_math_numeric_value", refactored.simple_math_numeric_value, "text"),
        ("parse_numeric_token", refactored.parse_numeric_token, "token"),
        ("normalize_gsm8k_gold", refactored.normalize_gsm8k_gold, "value"),
    ],
)
def test_single_argument_functions_match_original(
    original_answer_parsing: dict[str, Any],
    original_name: str,
    refactored_function: Any,
    keyword: str,
) -> None:
    original_function = original_answer_parsing[original_name]
    for text in TEXTS:
        expected = call_safely(original_function, text)
        actual = call_safely(refactored_function, **{keyword: text})
        assert actual == expected, f"{original_name}({text!r})"


def test_pair_functions_match_original(original_answer_parsing: dict[str, Any]) -> None:
    golds = list(HANDWRITTEN_GOLDS) + TEXTS[:200]
    for completion in TEXTS[:600]:
        for gold in golds[:: max(1, len(golds) // 25)]:
            assert call_safely(
                refactored.pissa_math_is_equiv,
                first=completion,
                second=gold,
            ) == call_safely(original_answer_parsing["pissa_math_is_equiv"], completion, gold)
            assert call_safely(
                refactored.robust_math_is_equiv,
                prediction=completion,
                gold=gold,
            ) == call_safely(original_answer_parsing["robust_math_is_equiv"], completion, gold)
            assert call_safely(
                refactored.process_pissa_math_result,
                completion=completion,
                answer=gold,
            ) == call_safely(original_answer_parsing["process_pissa_math_result"], completion, gold)


def original_gsm8k_check(namespace: dict[str, Any], completion: str, gold_raw: str) -> tuple[Any, ...]:
    gold = namespace["normalize_gsm8k_gold"](gold_raw)
    prediction, source = namespace["extract_robust_gsm8k_number"](completion)
    correct = prediction is not None and namespace["math"].isclose(
        float(prediction),
        float(gold),
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    return None if prediction is None else str(prediction), str(gold), bool(correct), source


def original_math_check(namespace: dict[str, Any], completion: str, gold_raw: str) -> tuple[Any, ...]:
    prediction, source = namespace["extract_robust_answer_text"](completion)
    correct = prediction is not None and namespace["robust_math_is_equiv"](prediction, gold_raw)
    return None if prediction is None else str(prediction), str(gold_raw), bool(correct), source


def test_answer_checkers_match_original_evaluation_branches(original_answer_parsing: dict[str, Any]) -> None:
    gsm8k_golds = ("72", "#### 1,234", "-5", "3.5")
    math_golds = HANDWRITTEN_GOLDS
    for completion in TEXTS[:1500]:
        for gold in gsm8k_golds:
            check = Gsm8kAnswerChecker().check(
                completion=completion,
                gold_raw=gold,
            )
            actual = (check.pred_answer, check.gold_answer, check.correct, check.extraction_source)
            assert actual == original_gsm8k_check(original_answer_parsing, completion, gold)
        for gold in math_golds:
            check = MathAnswerChecker().check(
                completion=completion,
                gold_raw=gold,
            )
            actual = (check.pred_answer, check.gold_answer, check.correct, check.extraction_source)
            assert actual == original_math_check(original_answer_parsing, completion, gold)
