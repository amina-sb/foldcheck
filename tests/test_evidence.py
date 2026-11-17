"""The README's evidence table must be exactly what scripts/evidence.py prints."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_evidence() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evidence", ROOT / "scripts" / "evidence.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PAGES = ["README.md", "docs/evidence.md"]


def _block(path: str, marker: str) -> str:
    text = (ROOT / path).read_text(encoding="utf-8")
    m = re.search(rf"<!-- {marker}:start -->\n(.*?)\n<!-- {marker}:end -->", text, re.S)
    assert m is not None, f"{path} is missing the {marker} markers"
    return m.group(1).strip()


@pytest.fixture(scope="module")
def evidence() -> ModuleType:
    return _load_evidence()


@pytest.mark.parametrize("path", PAGES)
def test_evidence_table_matches_script(path: str, evidence: ModuleType) -> None:
    assert _block(path, "evidence") == evidence.table().strip()


@pytest.fixture(scope="module")
def coverage(evidence: ModuleType) -> str:
    return str(evidence.coverage_table()).strip()


@pytest.mark.parametrize("path", PAGES)
def test_coverage_table_matches_script(path: str, coverage: str) -> None:
    assert _block(path, "coverage") == coverage
