from __future__ import annotations

import re


VALID_UNCERTAINTY = {"low", "medium", "high"}


def validate_uncertainty(value: str) -> str:
    normalized = (value or "").strip().lower()
    return normalized if normalized in VALID_UNCERTAINTY else "medium"


def is_valid_cell_id(value: str, rows: int = 4, cols: int = 4) -> bool:
    text = (value or "").strip().upper()
    match = re.fullmatch(r"([A-Z])([0-9]+)", text)
    if not match:
        return False
    row_idx = ord(match.group(1)) - ord("A")
    col_idx = int(match.group(2)) - 1
    return 0 <= row_idx < rows and 0 <= col_idx < cols
