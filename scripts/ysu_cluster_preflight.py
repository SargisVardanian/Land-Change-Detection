from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class PreflightResult:
    host: str
    ssh_alias: str
    conda_alias: str
    user: str
    terminal_target: str
    ssh_config_exists: bool
    ssh_alias_present: bool
    conda_alias_present: bool
    host_resolves: bool
    resolved_addresses: list[str]
    ssh_binary_present: bool
    rsync_binary_present: bool
    ssh_batch_reachable: bool
    preferred_ssh_target: str
    notes: list[str]

    @property
    def ready(self) -> bool:
        return (
            self.host_resolves
            and self.ssh_binary_present
            and self.rsync_binary_present
            and bool(self.user)
            and self.ssh_batch_reachable
        )


def inspect_ssh_config(path: Path, alias: str) -> tuple[bool, bool]:
    if not path.exists():
        return False, False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return True, f"Host {alias}" in text


def resolve_host(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return []
    return sorted({info[4][0] for info in infos})


def probe_ssh_batch(host_or_alias: str, timeout: int) -> bool:
    if shutil.which("ssh") is None:
        return False
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={timeout}",
            host_or_alias,
            "true",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def recommended_ssh_target(host: str, user: str, terminal_target: str) -> str:
    suffix = f"+{terminal_target}" if terminal_target else ""
    return f"{user}{suffix}@{host}" if user else host


def run_preflight(
    host: str,
    ssh_alias: str,
    conda_alias: str,
    user: str,
    terminal_target: str,
    ssh_config: Path,
    timeout: int,
) -> PreflightResult:
    ssh_config_exists, ssh_alias_present = inspect_ssh_config(ssh_config, ssh_alias)
    _, conda_alias_present = inspect_ssh_config(ssh_config, conda_alias)
    resolved = resolve_host(host)
    ssh_present = shutil.which("ssh") is not None
    rsync_present = shutil.which("rsync") is not None
    notes: list[str] = []
    if not ssh_config_exists:
        notes.append("~/.ssh/config is missing.")
    elif not ssh_alias_present:
        notes.append(f"SSH alias '{ssh_alias}' is not present in {ssh_config}.")
    if not conda_alias_present:
        notes.append(f"SSH alias '{conda_alias}' is not present in {ssh_config}.")
    if not resolved:
        notes.append(f"DNS did not return an address for {host}. VPN or campus DNS may be required.")
    if not ssh_present:
        notes.append("ssh is not installed or not on PATH.")
    if not rsync_present:
        notes.append("rsync is not installed or not on PATH.")
    if not user:
        notes.append("REMOTE_USER / --user was not provided.")
    ssh_target = conda_alias if conda_alias_present else recommended_ssh_target(host, user, terminal_target)
    reachable = probe_ssh_batch(ssh_target, timeout) if resolved else False
    if resolved and not reachable:
        notes.append("SSH batch probe failed. Key setup, password rotation, or VPN may still be needed.")
    return PreflightResult(
        host=host,
        ssh_alias=ssh_alias,
        conda_alias=conda_alias,
        user=user,
        terminal_target=terminal_target,
        ssh_config_exists=ssh_config_exists,
        ssh_alias_present=ssh_alias_present,
        conda_alias_present=conda_alias_present,
        host_resolves=bool(resolved),
        resolved_addresses=resolved,
        ssh_binary_present=ssh_present,
        rsync_binary_present=rsync_present,
        ssh_batch_reachable=reachable,
        preferred_ssh_target=ssh_target,
        notes=notes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether this MacBook is ready to push the project to YSU-HPC.")
    parser.add_argument("--host", default="cluster.ysu.am")
    parser.add_argument("--ssh-alias", default="ysu-hpc")
    parser.add_argument("--conda-alias", default="ysu-hpc-conda")
    parser.add_argument("--user", default="")
    parser.add_argument("--terminal-target", default="conda")
    parser.add_argument("--ssh-config", type=Path, default=Path.home() / ".ssh" / "config")
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="Emit the result as JSON.")
    args = parser.parse_args()

    result = run_preflight(
        args.host,
        args.ssh_alias,
        args.conda_alias,
        args.user,
        args.terminal_target,
        args.ssh_config,
        args.timeout,
    )
    if args.json:
        print(json.dumps(asdict(result) | {"ready": result.ready}, indent=2))
    else:
        print(f"host: {result.host}")
        print(f"ssh_alias: {result.ssh_alias}")
        print(f"ssh_config_exists: {result.ssh_config_exists}")
        print(f"ssh_alias_present: {result.ssh_alias_present}")
        print(f"host_resolves: {result.host_resolves}")
        print(f"resolved_addresses: {', '.join(result.resolved_addresses) if result.resolved_addresses else '-'}")
        print(f"ssh_binary_present: {result.ssh_binary_present}")
        print(f"rsync_binary_present: {result.rsync_binary_present}")
        print(f"ssh_batch_reachable: {result.ssh_batch_reachable}")
        print(f"ready: {result.ready}")
        if result.notes:
            print("notes:")
            for note in result.notes:
                print(f"- {note}")
    return 0 if result.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
