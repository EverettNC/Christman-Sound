# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# ==============================================================================
"""
Offline only. Nothing in the package downloads anything.

Ruling these tests are written to (Everett): the voice runs offline. No
outside engine, no browser speech, no downloads. A model is loaded from a
local path the caller gives, or the code raises with the reason.

The scan reads the real source with the `ast` module; it imports nothing it
scans. It covers christman_voice_sdk/, christman_sound/ and scripts/.
It fails on:
  - `from_pretrained(...)` without `local_files_only=True`
  - hf_hub_download, snapshot_download, cached_download, urlretrieve,
    urlopen, download_url_to_file, load_state_dict_from_url, torch.hub.load
  - importing huggingface_hub, requests, httpx, urllib3, aiohttp, wget, or
    urllib.request
  - constructing PaddleOCR, which fetches its model files on first use

christman_sound/core.py is outside the scan: Everett's standing ruling is
that it is never touched and never wired. It still calls from_pretrained
(core.py:371-372); nothing imports core.py, so that line never runs.

Two tables name what the scan still finds, so the suite stays honest while
other lanes finish:
  LOOPBACK_ONLY  network calls that only reach a server the same file
                 starts on 127.0.0.1. Not downloads. They stay.
  OPEN           downloads not yet removed, by file, with the most each file
                 may still hold and the owner who removes them. A file can
                 only go down from its count. Any new download anywhere
                 fails the suite. When an owner removes one, the count is
                 lowered or the entry deleted.
"""

from __future__ import annotations

import ast
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DOWNLOAD_CALLS = {
    "hf_hub_download", "snapshot_download", "cached_download", "urlretrieve",
    "urlopen", "download_url_to_file", "load_state_dict_from_url",
}
NETWORK_MODULES = {"huggingface_hub", "requests", "httpx", "urllib3", "aiohttp", "wget"}

#: file -> why its network use is not a download.
LOOPBACK_ONLY = {
    "scripts/verify_corti_wire.py": "posts cards to the corti_receiver it starts itself on 127.0.0.1",
}

#: file -> (most downloads allowed to remain, owner and what it fetches).
#: Keyed by file, not line, so an owner's edits elsewhere in the file do not
#: break this test. A file may only go down from its count, never up. When an
#: owner removes a download, lower the count or delete the entry.
OPEN: dict[str, tuple[int, str]] = {}


def _scanned_files() -> list[Path]:
    files = []
    for pkg in ("christman_voice_sdk", "christman_sound", "scripts"):
        files.extend(sorted((ROOT / pkg).rglob("*.py")))
    core = ROOT / "christman_sound" / "core.py"  # never touched, never wired
    return [f for f in files if f.is_file() and f != core]


def _call_name(func: ast.AST):
    if isinstance(func, ast.Attribute):
        return func.attr
    return getattr(func, "id", None)


def _downloads(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in NETWORK_MODULES or alias.name.startswith("urllib.request"):
                    found.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module.split(".")[0] in NETWORK_MODULES or node.module.startswith("urllib.request"):
                found.append((node.lineno, f"from {node.module} import"))
            if node.module == "urllib" and any(a.name == "request" for a in node.names):
                found.append((node.lineno, "from urllib import request"))
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name == "from_pretrained":
                local_only = any(
                    k.arg == "local_files_only" and isinstance(k.value, ast.Constant) and k.value.value is True
                    for k in node.keywords
                )
                if not local_only:
                    found.append((node.lineno, "from_pretrained without local_files_only=True"))
            elif name in DOWNLOAD_CALLS:
                found.append((node.lineno, f"{name}(...)"))
            elif name == "load" and isinstance(node.func, ast.Attribute) and _call_name(node.func.value) == "hub":
                found.append((node.lineno, "torch.hub.load(...)"))
            elif name == "PaddleOCR":
                found.append((node.lineno, "PaddleOCR(...) fetches its models on first construction"))
    return found


def _all_findings() -> dict[tuple[str, int], str]:
    findings = {}
    for path in _scanned_files():
        rel = str(path.relative_to(ROOT))
        if rel in LOOPBACK_ONLY:
            continue
        for line, what in _downloads(path):
            findings[(rel, line)] = what
    return findings


def test_nothing_in_the_package_downloads():
    files = _scanned_files()
    assert len(files) > 50, f"the scan found only {len(files)} files; the tree moved"
    findings = _all_findings()
    problems = []
    for rel in sorted({f for f, _ in findings}):
        here = sorted((line, what) for (f, line), what in findings.items() if f == rel)
        allowed = OPEN.get(rel, (0, ""))[0]
        if len(here) > allowed:
            problems.extend(f"{rel}:{line} {what}" for line, what in here)
    assert problems == [], "\n".join(problems)


def test_open_table_is_honest_about_what_remains():
    """Every OPEN entry has a count, an owner and a reason, and names a real file."""
    for rel, (count, why) in OPEN.items():
        assert count > 0 and why, rel
        assert (ROOT / rel).is_file(), f"{rel} is gone; delete its OPEN entry"


def test_the_scan_catches_each_kind_of_download(tmp_path):
    """The scanner itself: one sample of each pattern, and the allowed form."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import requests\n"
        "from huggingface_hub import hf_hub_download\n"
        "import urllib.request\n"
        "from urllib.parse import urlsplit\n"
        "M.from_pretrained('org/model')\n"
        "M.from_pretrained('/local/dir', local_files_only=True)\n"
        "hf_hub_download('org/model', 'f')\n"
        "torch.hub.load('x', 'y')\n"
        "PaddleOCR(lang='en')\n",
        encoding="utf-8",
    )
    lines = sorted(line for line, _ in _downloads(sample))
    assert lines == [1, 2, 3, 5, 7, 8, 9], lines


# ------------------------------------------------------------- Shorty, offline

def _no_network(monkeypatch):
    """Any attempt to open a network connection fails the test."""
    def refuse(*args, **kwargs):
        raise AssertionError(f"a network connection was attempted: {args!r}")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _shorty():
    for module in ("torch", "torchaudio", "transformers"):
        try:
            __import__(module)
        except ImportError:
            pytest.skip(f"not run: {module} is not installed here; shorty_emotion imports it at module top")
    from christman_voice_sdk.timbre import shorty_emotion
    return shorty_emotion


def test_shorty_with_no_paths_raises_and_fetches_nothing(monkeypatch):
    se = _shorty()
    _no_network(monkeypatch)
    calls = []
    monkeypatch.setattr(se.Wav2Vec2Model, "from_pretrained", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(se.Wav2Vec2Processor, "from_pretrained", lambda *a, **k: calls.append(a))
    with pytest.raises(se.ShortyModelMissing) as info:
        se.ShortyEmotionDetector()
    message = str(info.value)
    assert "no wav2vec_model_dir was given" in message and "no pca_model_path was given" in message
    assert calls == [], "a model load was attempted before the paths were checked"


def test_shorty_names_every_missing_file(monkeypatch, tmp_path):
    se = _shorty()
    _no_network(monkeypatch)
    model_dir = tmp_path / "wav2vec2"
    model_dir.mkdir()  # exists, but holds no saved model
    with pytest.raises(se.ShortyModelMissing) as info:
        se.ShortyEmotionDetector(
            wav2vec_model_dir=str(model_dir),
            pca_model_path=str(tmp_path / "pca.pt"),
            scaler_path=str(tmp_path / "scaler.pt"),
        )
    message = str(info.value)
    assert "has no config.json" in message
    assert "projection file is not at" in message and "scaler file is not at" in message


def test_shorty_projection_is_the_pca_math(tmp_path):
    se = _shorty()
    import numpy as np
    import torch

    rng = np.random.default_rng(7)
    n, d = len(se.SHORTY_EMOTION_LABELS), 24
    comps = rng.standard_normal((n, d))
    mean = rng.standard_normal(d)
    torch.save({"components": torch.tensor(comps), "mean": torch.tensor(mean)}, tmp_path / "pca.pt")
    s_mean, s_scale = rng.standard_normal(n), rng.uniform(0.5, 2.0, n)
    torch.save({"mean": torch.tensor(s_mean), "scale": torch.tensor(s_scale)}, tmp_path / "scaler.pt")

    proj = se.load_projection(tmp_path / "pca.pt")
    scaler = se.load_scaler(tmp_path / "scaler.pt")
    x = rng.standard_normal((1, d))
    expected = ((x - mean) @ comps.T - s_mean) / s_scale
    assert np.allclose(se.project(x, proj, scaler), expected)

    with pytest.raises(ValueError, match="does not match"):
        se.project(rng.standard_normal((1, d + 1)), proj)


def test_shorty_refuses_a_projection_that_leaves_a_label_unscored(tmp_path):
    se = _shorty()
    import torch

    n = len(se.SHORTY_EMOTION_LABELS)
    torch.save({"components": torch.zeros(n - 1, 8), "mean": torch.zeros(8)}, tmp_path / "short.pt")
    with pytest.raises(ValueError, match="One row per label"):
        se.load_projection(tmp_path / "short.pt")


def test_shorty_refuses_a_pickled_object(tmp_path):
    """weights_only=True stays on: a pickled object is refused, with the format named."""
    se = _shorty()
    import pickle

    (tmp_path / "pickled.pt").write_bytes(pickle.dumps({"components": object(), "mean": object()}))
    with pytest.raises(ValueError, match="weights_only=True"):
        se.load_projection(tmp_path / "pickled.pt")
