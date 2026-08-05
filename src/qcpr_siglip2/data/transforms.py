"""Deterministic synchronized geometry for temporal image pairs."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class SynchronizedResize:
    """Resize T1/T2 to one fixed footprint with no independent randomness."""

    size: tuple[int, int] = (256, 256)
    resampling: int = Image.Resampling.BICUBIC

    def __call__(
        self, t1: Image.Image, t2: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        if len(self.size) != 2 or min(self.size) <= 0:
            raise ValueError("resize size must contain two positive dimensions")
        return t1.resize(self.size, self.resampling), t2.resize(
            self.size, self.resampling
        )


def assert_pair_geometry(t1: Image.Image, t2: Image.Image) -> None:
    """Reject a pair whose synchronized path would silently use mismatched geometry."""

    if t1.mode != t2.mode:
        raise ValueError(f"temporal pair modes differ: {t1.mode!r} vs {t2.mode!r}")
