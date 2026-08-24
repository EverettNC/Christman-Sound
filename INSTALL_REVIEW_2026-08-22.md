# Christman-Sound install review — 2026-08-22 (Intel Mac x86_64, Python 3.12.12)

**Law:** Corti (`/Users/EverettN/CORTI`) and Christman-Sound are **separate repos**.
Sound only consumes Corti cards via `corti_ingest` / `corti_receiver` / `hearing`.
Do not merge the Corti app into this package.

## What works (proved)

```bash
# Giuseppe venv
/Users/EverettN/Giuseppe-Skyrider/venv/bin/python -m pip install -e /Users/EverettN/Christman-Sound --no-deps
```

Imports (after editable `--no-deps`):

- `import christman_voice_sdk` → OK
- `from christman_voice_sdk.audio import corti_receiver, corti_ingest, hearing` → OK
- `import christman_sound` → OK
- `import CHRISTMAN_EAR_CANAL` → OK
- `scripts/verify_corti_wire.py` → **PASS** (after origin fix below)

## Defects (with proof)

### 1. Hard `librosa` dep breaks full editable install on this machine

`pip install -e /Users/EverettN/Christman-Sound` (with deps) fails building **llvmlite**
(pulled by librosa → numba → llvmlite). No usable wheel; source build needs LLVM/cmake
and fails (`LLVM_DIR` / cmake configure incomplete). Exit code 1.

Workaround proved: `pip install -e … --no-deps`, then install non-librosa deps separately
(flask, pygame, pyttsx3, praat-parselmouth). Package metadata still *declares* librosa
Required, so pip warns about unresolved requires — honest conflict, not silent.

**Corti wire does not need librosa.** Treat librosa as optional for beings that only seat Hearing.

### 2. `setup.py` vs `pyproject.toml` torch pin drift

- `setup.py`: bare `"torch"`, `"torchaudio"` (no Intel Mac ceiling)
- `pyproject.toml`: `torch<2.3` / `torchaudio<2.3` on darwin x86_64

Modern pip uses `pyproject.toml` (good). `setup.py` is stale and dangerous if anyone
runs legacy `python setup.py install`.

### 3. `pyttsx3` → full `pyobjc` suite

Installing `pyttsx3` on macOS pulls essentially the entire pyobjc framework set
(100+ wheels). Valid for speak fallback; expensive and surprising for a Corti-wire-only being.

### 4. `verify_corti_wire.py` tested stale CORS origin `http://localhost:8080`

Receiver `DEFAULT_ALLOWED_ORIGINS` is **5487** only (Corti-in-the-being). Comments in
`corti_receiver.py` say 8080 is lab and does not post cards here. Verify script still
posted with Origin 8080 → **403**, false FAIL.

**Fixed 2026-08-22:** verify script now uses `http://localhost:5487`. Re-run → ALL PROVED.

CORTI repo (`package.json`): `being` = port 5487; `dev` = 8080. Keep them separate from Sound.

### 5. Giuseppe `requirements-sound.txt` lists `-e` path but SpeechRecognition was missing

Giuseppe listen path wants `SpeechRecognition>=3.10`. Install separately if not present.
Not a Sound packaging bug; Giuseppe install gap.

## Recommended install for Giuseppe / Cortee tests

```bash
cd /Users/EverettN/Giuseppe-Skyrider
./venv/bin/python -m pip install -e /Users/EverettN/Christman-Sound --no-deps
./venv/bin/python -m pip install flask pygame pyttsx3 praat-parselmouth 'SpeechRecognition>=3.10'
./venv/bin/python /Users/EverettN/Christman-Sound/scripts/verify_corti_wire.py
```

Do **not** copy `/Users/EverettN/CORTI` into Christman-Sound.
