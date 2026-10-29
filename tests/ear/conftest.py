# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""The ear's tests build its C library first (christman_sound/ear/build.py),
the way ear-core's run-tests.sh runs make before its tests. A build that
fails stops these tests with the compiler's own message; nothing is skipped
for it. Heavy runs: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 10.
"""

from christman_sound.ear import build as _ear_build

_ear_build.build()
