from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose MacBook-side access to YSU-HPC before bootstrap.")
    parser.add_argument("--host", default="cluster.ysu.am")
    parser.add_argument("--ssh-alias", default="ysu-hpc")
    parser.add_argument("--conda-alias", default="ysu-hpc-conda")
    parser.add_argument("--user", default="")
    parser.add_argument("--terminal-target", default="conda")
    parser.add_argument("--ssh-config", type=Path, default=Path.home() / ".ssh" / "config")
    parser.add_argument("--wireguard-dir", type=Path, default=Path("cluster/ysu/private/wireguard"))
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _run(cmd: list[str]) -> dict:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return {
        "cmd": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _resolve_host(host: str) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return []
    return sorted({info[4][0] for info in infos})


def _ssh_alias_present(ssh_config: Path, alias: str) -> bool:
    if not ssh_config.exists():
        return False
    return f"Host {alias}" in ssh_config.read_text(encoding="utf-8", errors="ignore")


def _wireguard_profiles(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(str(item) for item in path.glob("*.conf"))


def _recommended_ssh_target(host: str, user: str, terminal_target: str) -> str:
    if not user:
        return host
    suffix = f"+{terminal_target}" if terminal_target else ""
    return f"{user}{suffix}@{host}"


def build_report(
    host: str,
    ssh_alias: str,
    conda_alias: str,
    user: str,
    terminal_target: str,
    ssh_config: Path,
    wireguard_dir: Path,
    timeout: int,
) -> dict:
    resolved = _resolve_host(host)
    alias_present = _ssh_alias_present(ssh_config, ssh_alias)
    conda_alias_present = _ssh_alias_present(ssh_config, conda_alias)
    direct_target = _recommended_ssh_target(host, user, terminal_target)
    preferred_target = conda_alias if conda_alias_present else direct_target
    diagnostics = {
        "host": host,
        "resolved_addresses": resolved,
        "user": user,
        "terminal_target": terminal_target,
        "ssh_config_exists": ssh_config.exists(),
        "ssh_alias_present": alias_present,
        "conda_alias_present": conda_alias_present,
        "ssh_binary_present": shutil.which("ssh") is not None,
        "rsync_binary_present": shutil.which("rsync") is not None,
        "curl_binary_present": shutil.which("curl") is not None,
        "dig_binary_present": shutil.which("dig") is not None,
        "nslookup_binary_present": shutil.which("nslookup") is not None,
        "wireguard_profiles": _wireguard_profiles(wireguard_dir),
        "preferred_ssh_target": preferred_target,
        "recommended_direct_ssh_target": direct_target,
        "dns_probe": _run(["dig", "+short", host]) if shutil.which("dig") else None,
        "nslookup_probe": _run(["nslookup", host]) if shutil.which("nslookup") else None,
        "https_probe": _run(["curl", "-I", "--max-time", str(timeout), f"https://{host}"]) if shutil.which("curl") else None,
    }
    ssh_target = preferred_target
    if diagnostics["ssh_binary_present"]:
        diagnostics["ssh_probe"] = _run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={timeout}",
                ssh_target,
                "true",
            ]
        )
    else:
        diagnostics["ssh_probe"] = None

    notes: list[str] = []
    if not resolved:
        notes.append("DNS resolution failed for cluster host.")
    if diagnostics["wireguard_profiles"]:
        notes.append("WireGuard profiles are present in cluster/ysu/private/wireguard.")
    if not diagnostics["ssh_config_exists"]:
        notes.append("~/.ssh/config is missing.")
    elif not diagnostics["ssh_alias_present"]:
        notes.append(f"SSH alias '{ssh_alias}' is missing from ~/.ssh/config.")
    if not diagnostics["conda_alias_present"]:
        notes.append(f"SSH alias '{conda_alias}' is missing from ~/.ssh/config.")
    if not user:
        notes.append("REMOTE_USER / --user was not provided, so direct SSH-to-container validation is incomplete.")
    https_ok = diagnostics["https_probe"] and diagnostics["https_probe"]["returncode"] == 0
    if not https_ok:
        notes.append("HTTPS probe to the cluster portal failed.")
    ssh_ok = diagnostics["ssh_probe"] and diagnostics["ssh_probe"]["returncode"] == 0
    if not ssh_ok:
        notes.append("SSH batch probe failed.")

    diagnostics["notes"] = notes
    diagnostics["ready_for_bootstrap"] = bool(resolved) and bool(https_ok) and bool(ssh_ok) and bool(user)
    return diagnostics


def main() -> int:
    args = parse_args()
    report = build_report(
        args.host,
        args.ssh_alias,
        args.conda_alias,
        args.user,
        args.terminal_target,
        args.ssh_config,
        args.wireguard_dir,
        args.timeout,
    )
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["ready_for_bootstrap"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
