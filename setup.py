# WHAT CHANGED AND WHY (2026-10-17) — OUTSIDE VOICES CUT
# Everett's ruling: every outside speech engine is cut. Removed from
# install_requires:
#         "pyttsx3",
# pyttsx3 is the operating system's speech engine; nothing in the package
# imports it any more. (pygame was removed on 2026-10-13. Coqui TTS was never
# a declared dependency here.)
from setuptools import setup, find_packages

setup(
    name="christman-sound",
    version="1.0.0",
    author="Everett Nathaniel Christman",
    author_email="lumacognify@thechristmanaiproject.com",
    description="The Christman Voice SDK — complete voice intelligence package",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # numpy<2 required: torch<2.3 wheels (the ceiling on Intel Mac, where
        # PyTorch stopped publishing builds after 2.2.2) were compiled against
        # NumPy 1.x and crash under NumPy 2.x ("_ARRAY_API not found").
        "numpy<2",
        "librosa",
        "soundfile",
        "torch",
        "torchaudio",
        # transformers>=5 hard-requires torch>=2.4, which has no wheel for
        # Intel Mac (torch tops out at 2.2.2 there) — pin <5 so the torch
        # backend stays enabled instead of silently disabling itself.
        "transformers>=4.40,<5",
        "flask",
        "fastapi",
        "rich",
        "praat-parselmouth",  # PyPI name is praat-parselmouth, not parselmouth
    ],
)
