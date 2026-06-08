from __future__ import annotations

import argparse
from pathlib import Path


def render_host_block(alias: str, host: str, user: str) -> str:
    return (
        f"Host {alias}\n"
        f"    HostName {host}\n"
        f"    User {user}\n"
        "    ServerAliveInterval 60\n"
        "    ServerAliveCountMax 10\n"
    )


def upsert_host_block(config_text: str, alias: str, block: str) -> str:
    lines = config_text.splitlines()
    out: list[str] = []
    i = 0
    replaced = False
    while i < len(lines):
        line = lines[i]
        if line.strip() == f"Host {alias}":
            replaced = True
            out.extend(block.rstrip("\n").splitlines())
            i += 1
            while i < len(lines) and not lines[i].startswith("Host "):
                i += 1
            continue
        out.append(line)
        i += 1
    if not replaced:
        if out and out[-1] != "":
            out.append("")
        out.extend(block.rstrip("\n").splitlines())
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Write or update the YSU-HPC SSH alias in ~/.ssh/config.")
    parser.add_argument("--alias", default="ysu-hpc")
    parser.add_argument("--host", default="cluster.ysu.am")
    parser.add_argument("--user", required=True)
    parser.add_argument("--ssh-config", type=Path, default=Path.home() / ".ssh" / "config")
    args = parser.parse_args()

    args.ssh_config.parent.mkdir(parents=True, exist_ok=True)
    current = args.ssh_config.read_text(encoding="utf-8") if args.ssh_config.exists() else ""
    updated = upsert_host_block(current, args.alias, render_host_block(args.alias, args.host, args.user))
    args.ssh_config.write_text(updated, encoding="utf-8")
    print(f"Wrote SSH alias '{args.alias}' to {args.ssh_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
