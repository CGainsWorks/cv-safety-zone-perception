"""Shared fixtures.

``sys.path`` is prepended rather than relying on an installed package so that
``pytest`` works in a bare checkout with no build step.  ``make install`` does
install the package properly; this is for the case where somebody has cloned
the repository and wants to run the tests before reading anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from szp.data.synthetic import SceneConfig, SyntheticScene  # noqa: E402
from szp.detection.base import default_model_path  # noqa: E402
from szp.zones.config import load_zones  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def zone_set():
    return load_zones(REPO_ROOT / "config" / "zones.yaml")


@pytest.fixture(scope="session")
def scene() -> SyntheticScene:
    """A short scene, shared across tests.  Rendering is not free."""
    return SyntheticScene(SceneConfig(duration_s=6.0, fps=10.0))


@pytest.fixture(scope="session")
def model_path() -> Path:
    path = default_model_path()
    if not path.exists():
        pytest.skip(f"no detector weights at {path}; run scripts/fetch_model.py")
    return path


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: takes more than a couple of seconds")
