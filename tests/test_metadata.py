from __future__ import annotations

import sys
from pathlib import Path

import foldcheck

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    tomllib = None


def test_version_matches_pyproject() -> None:
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    if tomllib is not None:
        version = tomllib.loads(text)["project"]["version"]
    else:
        line = next(ln for ln in text.splitlines() if ln.startswith("version = "))
        version = line.split('"')[1]
    assert foldcheck.__version__ == version


def test_public_names_resolve() -> None:
    for name in foldcheck.__all__:
        assert hasattr(foldcheck, name), name
