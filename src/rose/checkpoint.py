from __future__ import annotations

import logging
from typing import Any

from torch import nn

_LOG_TRUNC = 40


def load_state_dict_relaxed(
    model: nn.Module,
    state_dict: dict[str, Any],
    *,
    logger: logging.Logger | None = None,
    tag: str = "checkpoint",
) -> tuple[list[str], list[str]]:
    incomp = model.load_state_dict(state_dict, strict=False)
    missing = list(incomp.missing_keys)
    unexpected = list(incomp.unexpected_keys)
    if logger is None:
        return (missing, unexpected)

    def _fmt(keys: list[str]) -> str:
        if len(keys) <= _LOG_TRUNC:
            return str(keys)
        return str(keys[:_LOG_TRUNC]) + f" ... (+{len(keys) - _LOG_TRUNC} more)"

    if missing:
        logger.warning("[%s] missing_keys (%d): %s", tag, len(missing), _fmt(missing))
    if unexpected:
        logger.warning("[%s] unexpected_keys (%d): %s", tag, len(unexpected), _fmt(unexpected))
    return (missing, unexpected)
