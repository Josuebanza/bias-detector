"""Scoring helpers.

The main scoring logic lives in analytics.bias_rules (fast + deterministic).
This module exists to make future refactors easy.
"""

from .bias_rules import _clamp, _level

__all__ = ["_clamp", "_level"]
