from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class YSUClusterLayout:
    repo_root: Path
    cluster_user: str = "svardanyan"
    project_slug: str = "land-change-detection"
    weka_base: Path | None = None
    home_base: Path | None = None
    conda_env_name: str = "lcd"

    @property
    def weka_root(self) -> Path:
        if self.weka_base is not None:
            return self.weka_base
        return Path("/mnt/weka") / self.cluster_user / self.project_slug

    @property
    def home_root(self) -> Path:
        if self.home_base is not None:
            return self.home_base
        return Path("/home") / self.cluster_user

    @property
    def data_root(self) -> Path:
        return self.weka_root / "data"

    @property
    def artifacts_root(self) -> Path:
        return self.weka_root / "artifacts"

    @property
    def slurm_logs_root(self) -> Path:
        return self.weka_root / "slurm_logs"

    @property
    def runs_root(self) -> Path:
        return self.weka_root / "runs"

    @property
    def conda_base(self) -> Path:
        return self.home_root / "miniconda3"

    @property
    def repo_data_link(self) -> Path:
        return self.repo_root / "data"

    @property
    def repo_artifacts_link(self) -> Path:
        return self.repo_root / "artifacts"

    def managed_directories(self) -> list[Path]:
        return [
            self.weka_root,
            self.data_root,
            self.artifacts_root,
            self.slurm_logs_root,
            self.runs_root,
        ]

    def symlink_pairs(self) -> list[tuple[Path, Path]]:
        return [
            (self.repo_data_link, self.data_root),
            (self.repo_artifacts_link, self.artifacts_root),
        ]

    def shell_exports(self) -> dict[str, str]:
        return {
            "LCDC_CLUSTER_USER": self.cluster_user,
            "LCDC_PROJECT_SLUG": self.project_slug,
            "LCDC_REPO_ROOT": str(self.repo_root),
            "LCDC_WEKA_ROOT": str(self.weka_root),
            "LCDC_DATA_ROOT": str(self.data_root),
            "LCDC_ARTIFACTS_ROOT": str(self.artifacts_root),
            "LCDC_SLURM_LOGS_ROOT": str(self.slurm_logs_root),
            "LCDC_RUNS_ROOT": str(self.runs_root),
            "LCDC_HOME_ROOT": str(self.home_root),
            "LCDC_CONDA_BASE": str(self.conda_base),
            "LCDC_CONDA_ENV": self.conda_env_name,
        }


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_symlink(link_path: Path, target_path: Path, *, force: bool = False) -> None:
    if link_path.is_symlink():
        if Path(link_path.resolve()) == target_path.resolve():
            return
        if not force:
            raise FileExistsError(f"{link_path} already points to {link_path.resolve()}, not {target_path}")
        link_path.unlink()
    elif link_path.exists():
        if not force:
            raise FileExistsError(f"{link_path} exists and is not a symlink")
        if link_path.is_dir():
            raise IsADirectoryError(f"{link_path} exists as a directory; move it before recreating the symlink")
        raise FileExistsError(f"{link_path} exists and is not a symlink")

    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target_path)
