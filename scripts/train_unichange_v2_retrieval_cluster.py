from __future__ import annotations

import sys
from pathlib import Path

import train_unichange_v2_retrieval as base
from ucv2_cluster_common import build_model, strict_device
from ucv2_cluster_report import finalize


def arg_value(name: str, default: str = "") -> str:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main() -> int:
    device_name = arg_value("--device", "cuda")
    strict_device(device_name)
    if "--smoke" in sys.argv and "--epochs" not in sys.argv:
        sys.argv.extend(["--epochs", "2"])
    base._build_model = build_model
    result = base.main()
    report = finalize(Path(arg_value("--output-dir")), device_name)
    if "--smoke" in sys.argv and not report.get("fake_backbones") and not report["real_cluster_smoke_passed"]:
        raise RuntimeError("Real cluster smoke failed; inspect smoke_report.json")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
