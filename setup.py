# WHAT CHANGED AND WHY (2026-10-28, Lane 6)
# This file used to carry its own dependency list. pyproject.toml carries one
# too, and setuptools reads pyproject.toml's [project] table, so the two lists
# drifted apart (this one had no Intel Mac markers for torch, and both carried
# librosa, which nothing imports and which broke the install on Intel Mac).
# Everything now lives in pyproject.toml. This shim stays so tools that still
# call setup.py work.
#
# setup.py does NOT build the C library, and never did. Build it with
#
#     python3 scripts/build_dsp.py --check
#
# which compiles christman_dsp.c and christman_synth.c into
# christman_sound/christman_dsp.so, the file christman_sound/dsp.py loads.
#
# 2026-10-17: pyttsx3 was removed (Everett's ruling cuts every outside speech
# engine). pygame was removed on 2026-10-13.
from setuptools import setup

setup()
