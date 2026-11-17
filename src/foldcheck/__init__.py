"""Validation protocols for applied machine learning."""

from foldcheck.audit import (
    AuditReport,
    Finding,
    LeakageError,
    audit_cv,
    detect_global_scaling,
    scan_target_leakage,
)
from foldcheck.splitters import GroupKFold, TimeSeriesSplit

__version__ = "0.1.0"

__all__ = [
    "AuditReport",
    "Finding",
    "GroupKFold",
    "LeakageError",
    "TimeSeriesSplit",
    "__version__",
    "audit_cv",
    "detect_global_scaling",
    "scan_target_leakage",
]
