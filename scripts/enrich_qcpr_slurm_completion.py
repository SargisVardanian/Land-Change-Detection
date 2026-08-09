#!/usr/bin/env python3
"""Enrich a compute-produced completion snapshot from a Slurm login node."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--sacct-bin", default="sacct")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root)
    completion_path = run_root / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    job_id = str(completion["upstream_job_id"])
    output = subprocess.check_output(
        [
            args.sacct_bin,
            "-X",
            "-j",
            job_id,
            "--format=JobIDRaw,State,ExitCode,Elapsed,NodeList",
            "-n",
            "-P",
        ],
        text=True,
    )
    exact = [
        row
        for row in (line.split("|") for line in output.splitlines() if line.strip())
        if row[0] == job_id
    ]
    if len(exact) != 1 or len(exact[0]) < 5:
        raise RuntimeError(f"cannot resolve one exact sacct row for {job_id}")
    row = exact[0]
    payload = {
        "schema_version": "qcpr-slurm-accounting-enrichment-v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "completion_schema_version": completion["schema_version"],
        "upstream_job_id": job_id,
        "state": row[1],
        "exit_code": row[2],
        "elapsed": row[3],
        "node_list": row[4],
        "accounting_status": "RESOLVED_ON_LOGIN",
    }
    target = run_root / "completion_accounting.json"
    temporary = run_root / "completion_accounting.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
