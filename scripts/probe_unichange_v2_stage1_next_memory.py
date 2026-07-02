from __future__ import annotations

import sys
from pathlib import Path

from ucv2_stage1_next_memory_core import run


if __name__ == "__main__":
    raise SystemExit(
        run(
            Path(sys.argv[1]),
            Path(sys.argv[2]),
            Path(sys.argv[3]),
            Path(sys.argv[4]),
            Path(sys.argv[5]),
        )
    )
