"""The README's evidence table must be exactly what scripts/evidence.py prints."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def _load_evidence() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evidence", ROOT / "scripts" / "evidence.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_readme_table_matches_evidence_script() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"<!-- evidence:start -->\n(.*?)\n<!-- evidence:end -->", readme, re.S)
    assert m is not None, "README is missing the evidence markers"
    assert m.group(1).strip() == _load_evidence().table().strip()


def test_readme_coverage_table_matches_evidence_script() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    m = re.search(r"<!-- coverage:start -->\n(.*?)\n<!-- coverage:end -->", readme, re.S)
    assert m is not None, "README is missing the coverage markers"
    assert m.group(1).strip() == _load_evidence().coverage_table().strip()
