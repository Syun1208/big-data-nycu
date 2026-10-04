from __future__ import annotations

import math
import re
from fractions import Fraction

PISSA_ANSWER_MARKER = "The answer is: "
BOXED_MARKER = "\\boxed{"
HASH_MARKER = "####"
NUMERIC_ABSOLUTE_TOLERANCE = 1e-9

PISSA_GSM8K_NUMBER_PATTERN = re.compile(r"[\-+]?\d*[\.,/]?\d+")
GSM8K_GOLD_PATTERN = re.compile(r"[\-+]?\d+(?:\.\d+)?")
ANSWER_MARKER_PATTERN = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer"
    r"\s*(?:is\s*)?"
    r"(?::|=)?"
    r"\s*([^\n]+)",
    flags=re.IGNORECASE,
)
ROBUST_NUMBER_PATTERN = re.compile(
    r"[-+]?"
    r"(?:\d{1,3}(?:,\d{3})+|\d+|\.\d+)"
    r"(?:\.\d+)?"
    r"(?:/\d+(?:\.\d+)?)?"
)
NUMBER_PATTERN = r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))"
LATEX_FRACTION_PATTERN = re.compile(r"\\frac\{" + NUMBER_PATTERN + r"\}\{" + NUMBER_PATTERN + r"\}")
SLASH_FRACTION_PATTERN = re.compile(NUMBER_PATTERN + r"/" + NUMBER_PATTERN)


def remove_right_units(*, string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    return string


def fix_sqrt(*, string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            new_substr = "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def fix_fracs(*, string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) <= 1:
        return new_str

    for substr in substrs[1:]:
        new_str += "\\frac"
        if substr[0] == "{":
            new_str += substr
            continue
        if len(substr) < 2:
            return string
        numerator = substr[0]
        denominator = substr[1]
        post_substr = substr[2:]
        if denominator != "{":
            new_str += "{" + numerator + "}{" + denominator + "}" + post_substr
        else:
            new_str += "{" + numerator + "}" + denominator + post_substr
    return new_str


def fix_a_slash_b(*, string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    numerator, denominator = string.split("/")
    try:
        numerator_int = int(numerator)
        denominator_int = int(denominator)
        assert string == f"{numerator_int}/{denominator_int}"
        return "\\frac{" + str(numerator_int) + "}{" + str(denominator_int) + "}"
    except (AssertionError, ValueError):
        return string


def strip_pissa_math_string(*, string: object) -> str:
    text = str(string)
    for old, new in (
        ("\n", ""),
        ("\\!", ""),
        ("\\\\", "\\"),
        ("tfrac", "frac"),
        ("dfrac", "frac"),
        ("\\left", ""),
        ("\\right", ""),
        ("^{\\circ}", ""),
        ("^\\circ", ""),
        ("\\$", ""),
    ):
        text = text.replace(old, new)
    text = remove_right_units(string=text)
    for old, new in (
        ("\\%", ""),
        ("\\%", ""),
        (" .", " 0."),
        ("{.", "{0."),
    ):
        text = text.replace(old, new)

    if len(text) == 0:
        return text
    if text[0] == ".":
        text = "0" + text

    if len(text.split("=")) == 2 and len(text.split("=")[0]) <= 2:
        text = text.split("=")[1]

    text = fix_sqrt(string=text)
    text = text.replace(" ", "")
    text = fix_fracs(string=text)

    if text == "0.5":
        text = "\\frac{1}{2}"

    return fix_a_slash_b(string=text)


def pissa_math_is_equiv(
    *,
    first: str | None,
    second: str | None,
) -> bool:
    if first is None and second is None:
        return True
    if first is None or second is None:
        return False
    try:
        return strip_pissa_math_string(string=first) == strip_pissa_math_string(string=second)
    except Exception:
        return first == second


def process_pissa_math_result(
    *,
    completion: object,
    answer: object,
) -> bool:
    split_answer = str(completion).split(PISSA_ANSWER_MARKER)
    if len(split_answer) <= 1:
        return False

    extracted = split_answer[-1].split(".\n")[0].strip()
    if len(extracted) > 0 and extracted[-1] == ".":
        extracted = extracted[:-1]
    return pissa_math_is_equiv(
        first=extracted.strip(),
        second=str(answer),
    )


def is_number(*, text: str) -> bool:
    try:
        float(text)
        return True
    except Exception:
        return False


def extract_pissa_gsm8k_number(*, completion: object) -> int | None:
    parts = str(completion).split(PISSA_ANSWER_MARKER)
    if len(parts) <= 1:
        return None
    match = PISSA_GSM8K_NUMBER_PATTERN.search(parts[-1].strip())
    if not match:
        return None
    token = match.group().replace(",", "")

    if "/" in token:
        numerator, denominator = token.split("/", 1)
        if not (is_number(text=numerator) and is_number(text=denominator)):
            return None
        if float(denominator) == 0:
            return round(float(numerator))
        return round(float(Fraction(token)))

    value = float(token)
    if math.isinf(value):
        return None
    return round(value)


def normalize_gsm8k_gold(*, value: object) -> float:
    text = str(value).strip().replace(",", "")
    if HASH_MARKER in text:
        text = text.split(HASH_MARKER)[-1].strip()
    match = GSM8K_GOLD_PATTERN.search(text)
    if not match:
        raise ValueError(f"Cannot parse GSM8K gold: {value!r}")
    return float(match.group())


def extract_last_boxed(*, text: object) -> str | None:
    content = str(text)
    start = content.rfind(BOXED_MARKER)
    if start == -1:
        return None

    body_start = start + len(BOXED_MARKER)
    depth = 1
    for position in range(body_start, len(content)):
        if content[position] == "{":
            depth += 1
        elif content[position] == "}":
            depth -= 1
            if depth == 0:
                return content[body_start:position].strip()
    return None


def clean_robust_answer_text(*, text: object) -> str:
    cleaned = str(text).strip().strip("$")
    for wrapper in ("\\(", "\\)", "\\[", "\\]"):
        cleaned = cleaned.replace(wrapper, "")
    cleaned = cleaned.strip()

    while cleaned.endswith((".", ",", ";")):
        cleaned = cleaned[:-1].strip()
    return cleaned


def extract_robust_answer_text(*, completion: object) -> tuple[str | None, str | None]:
    text = str(completion)

    matches = list(ANSWER_MARKER_PATTERN.finditer(text))
    if matches:
        candidate = matches[-1].group(1).strip()
        boxed = extract_last_boxed(text=candidate)
        if boxed is not None:
            return clean_robust_answer_text(text=boxed), "answer_marker_boxed"
        return clean_robust_answer_text(text=candidate), "answer_marker"

    if HASH_MARKER in text:
        candidate = text.rsplit(HASH_MARKER, 1)[-1].strip()
        if candidate:
            candidate = candidate.splitlines()[0].strip()
            if candidate:
                return clean_robust_answer_text(text=candidate), "hash_marker"

    boxed = extract_last_boxed(text=text)
    if boxed is not None:
        return clean_robust_answer_text(text=boxed), "boxed"

    return None, None


def parse_numeric_token(*, token: object) -> float | None:
    if token is None:
        return None

    text = str(token).strip().replace(",", "")
    try:
        if "/" in text:
            return float(Fraction(text))
        return float(text)
    except Exception:
        return None


def extract_robust_gsm8k_number(*, completion: object) -> tuple[float | None, str]:
    candidate, source = extract_robust_answer_text(completion=completion)
    if candidate is None:
        candidate = str(completion)
        source = "full_text_fallback"

    matches = ROBUST_NUMBER_PATTERN.findall(candidate)
    if not matches:
        matches = ROBUST_NUMBER_PATTERN.findall(str(completion))
        source = "last_number_fallback"

    if not matches:
        return None, source
    return parse_numeric_token(token=matches[-1]), source


def simple_math_numeric_value(*, text: str | None) -> float | None:
    if text is None:
        return None

    cleaned = clean_robust_answer_text(text=text).replace(",", "")

    for pattern in (LATEX_FRACTION_PATTERN, SLASH_FRACTION_PATTERN):
        match = pattern.fullmatch(cleaned)
        if match:
            numerator = float(match.group(1))
            denominator = float(match.group(2))
            if denominator == 0:
                return None
            return numerator / denominator

    try:
        return float(cleaned)
    except Exception:
        return None


def robust_math_is_equiv(
    *,
    prediction: str | None,
    gold: str | None,
) -> bool:
    if prediction is None or gold is None:
        return False

    cleaned_prediction = clean_robust_answer_text(text=prediction)
    cleaned_gold = clean_robust_answer_text(text=gold)

    if pissa_math_is_equiv(
        first=cleaned_prediction,
        second=cleaned_gold,
    ):
        return True

    prediction_value = simple_math_numeric_value(text=cleaned_prediction)
    gold_value = simple_math_numeric_value(text=cleaned_gold)
    if prediction_value is None or gold_value is None:
        return False
    return math.isclose(
        prediction_value,
        gold_value,
        rel_tol=0.0,
        abs_tol=NUMERIC_ABSOLUTE_TOLERANCE,
    )
