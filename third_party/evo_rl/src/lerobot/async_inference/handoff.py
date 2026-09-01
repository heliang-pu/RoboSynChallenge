"""Compatibility imports for the reusable handoff skill.

New code should import from robot_skills.handoff. This module remains so
existing async-inference clients and command lines keep working unchanged.
"""

from robot_skills.handoff import (
    HandoffStabilityConfig,
    HandoffStabilityDetector,
    extract_handoff_joints,
)

__all__ = ["HandoffStabilityConfig", "HandoffStabilityDetector", "extract_handoff_joints"]
