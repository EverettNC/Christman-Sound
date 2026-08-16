# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
Back-compat shim. The monolith moved to christman_sound/core.py (2026-08-16)
so installed consumers can `from christman_sound.core import ...`.

Anything that still does `import core` from the repo root gets the same
module, not a copy.
"""

from christman_sound.core import *  # noqa: F401,F403
from christman_sound import core as _core

__doc__ = _core.__doc__
