from __future__ import annotations

import logging
import os
from collections.abc import Sequence

from src.interface.token_source import TokenSource

logger = logging.getLogger(__name__)

HF_TOKEN_NAME = "HF_TOKEN"


class EnvironmentTokenSource:
    def __init__(self, *, variable: str = HF_TOKEN_NAME) -> None:
        self._variable = variable

    def read_token(self) -> str | None:
        return os.environ.get(self._variable) or None


class KaggleSecretTokenSource:
    def __init__(self, *, label: str = HF_TOKEN_NAME) -> None:
        self._label = label

    def read_token(self) -> str | None:
        try:
            from kaggle_secrets import UserSecretsClient
        except ImportError:
            return None
        try:
            return UserSecretsClient().get_secret(self._label)
        except Exception:
            logger.debug("Kaggle secret %s unavailable", self._label, exc_info=True)
            return None


class FallbackTokenSource:
    def __init__(self, *, sources: Sequence[TokenSource]) -> None:
        self._sources = tuple(sources)

    def read_token(self) -> str | None:
        for source in self._sources:
            token = source.read_token()
            if token:
                logger.debug("HF token found via %s", type(source).__name__)
                return token
        return None


def login_huggingface(*, token_source: TokenSource) -> bool:
    from huggingface_hub import login

    token = token_source.read_token()
    if token is None:
        logger.warning("HF_TOKEN not found; continuing with cached credentials if any.")
        return False
    try:
        login(token=token)
    except Exception:
        logger.warning("Hugging Face login failed; continuing with cached credentials if any.", exc_info=True)
        return False
    logger.info("Hugging Face login: OK")
    return True


def build_token_source() -> TokenSource:
    return FallbackTokenSource(
        sources=(
            EnvironmentTokenSource(),
            KaggleSecretTokenSource(),
        )
    )
