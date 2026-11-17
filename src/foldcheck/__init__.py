"""Validation protocols for applied machine learning."""

from foldcheck.audit import (
    AuditReport,
    Finding,
    LeakageError,
    audit_cv,
    detect_global_scaling,
    scan_target_leakage,
)
from foldcheck.inference import (
    Interval,
    NestedCVResult,
    RepeatedCVResult,
    bootstrap_ci,
    corrected_t_interval,
    nested_cv,
    repeated_cv,
)
from foldcheck.splitters import GroupKFold, TimeSeriesSplit

__version__ = "0.1.0"

__all__ = [
    "AuditReport",
    "Finding",
    "GroupKFold",
    "Interval",
    "LeakageError",
    "NestedCVResult",
    "RepeatedCVResult",
    "TimeSeriesSplit",
    "__version__",
    "audit_cv",
    "bootstrap_ci",
    "corrected_t_interval",
    "detect_global_scaling",
    "nested_cv",
    "repeated_cv",
    "scan_target_leakage",
]
