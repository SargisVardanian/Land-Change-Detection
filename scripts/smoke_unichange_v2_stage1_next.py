from __future__ import annotations

import sys
from pathlib import Path

from ucv2_cluster_report import finalize
from ucv2_stage1_next_smoke_core import run


if __name__ == "__main__":
    output_dir = Path(sys.argv[2])
    result = run(
        Path(sys.argv[1]),
        output_dir,
        Path(sys.argv[3]),
        Path(sys.argv[4]),
        Path(sys.argv[5]),
    )
    report = finalize(output_dir, "cuda")
    required = (
        report.get("real_cluster_smoke_passed")
        and report.get("stage1_next") is True
        and report.get("loss") == "multi_positive_set_info_nce"
        and report.get("stable_caption_groups") is True
        and report.get("use_direction_embeddings") is True
        and report.get("use_explicit_change_fusion") is True
        and report.get("trainable_temperature") is True
    )
    if not required:
        raise SystemExit("Stage-1-next smoke report did not pass all feature gates")
    raise SystemExit(result)
