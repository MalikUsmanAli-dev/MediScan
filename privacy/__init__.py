"""
privacy — MediScan's Privacy Engine.

Confidence-driven, fully local privacy protection that runs BEFORE any
cloud AI provider is called. See privacy/engine.py for the pipeline.

Public API:
    PrivacyEngine, PrivacyConfig, PrivacyReport, DetectedRegion, FusedRegion
"""

from privacy.types import PrivacyConfig, PrivacyReport, DetectedRegion, FusedRegion
from privacy.engine import PrivacyEngine, FIELD_LABELS, ALL_FIELD_KEYS, DEFAULT_ENABLED_FIELDS, PII_FIELD_META, REDACTION_METHODS

__all__ = [
    "PrivacyEngine",
    "PrivacyConfig",
    "PrivacyReport",
    "DetectedRegion",
    "FusedRegion",
    "FIELD_LABELS",
    "ALL_FIELD_KEYS",
    "DEFAULT_ENABLED_FIELDS",
    "PII_FIELD_META",
    "REDACTION_METHODS",
]
