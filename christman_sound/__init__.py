# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
christman_sound — the package beings import, and where the voice lives.

The voice is here, in plain Python and C:
  - dsp.py          the one loader for christman_dsp.so (built by
                    scripts/build_dsp.py from christman_dsp.c,
                    christman_synth.c and christman_vocoder.c)
  - synth.py        synthesize() and speak(): text -> phones -> sound
  - stitch.py       frame voices, spoken from a voice's own recordings
  - lexicon.py      words -> phones
  - formants.py     formant measurement from recordings
  - voice_profile.py  a voice's measured profile
  - logger.py       logging

The package has no voice of its own: a being passes its own voice (a voice
profile file it measured with scripts/measure_voice.py, or a frame voice
folder) to synth.synthesize() and speak().

`christman_sound.core` is kept for the record. Everett's standing ruling: it
is called by nothing, is never wired into anything, and is not touched.

The modular SDK is `christman_voice_sdk`: hearing, recognition, tone, timbre,
the speaking engine (synthesis/voice_synthesis.py, which renders through
christman_sound.synth), and the rest.
"""
