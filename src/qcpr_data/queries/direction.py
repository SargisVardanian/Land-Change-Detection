"""Conservative temporal-direction extraction from already-written text."""

from __future__ import annotations

import re

FORWARD = re.compile(r"\b(appeared|constructed|built|expanded|increased|flooded|grew|added)\b", re.I)
REVERSE = re.compile(r"\b(disappeared|demolished|removed|contracted|decreased|receded|shrunk|lost)\b", re.I)


def infer_direction(text: str) -> str:
    forward = bool(FORWARD.search(text))
    reverse = bool(REVERSE.search(text))
    if forward and not reverse:
        return "forward"
    if reverse and not forward:
        return "reverse"
    return "none"
