"""
Voice diagnostics.

Generates synthesis samples across configured voice profiles and reports which
configurations produce identical audio — the actual diagnostic value: proving a
"regional variant" is a real variant and not the same file twice.

WHAT CHANGED AND WHY
--------------------

1. IT SENT THE USER'S TEXT TO GOOGLE. FOURTEEN TIMES.

       from gtts import gTTS
       tts = gTTS(text=text, lang=language, slow=slow_mode, tld=region_code)

   `gtts` is a cloud service. Seven regions x two speeds meant fourteen HTTP
   requests per diagnostic run, each carrying the test text off the machine.
   REMEDIATION Phase 3 lists removing `gtts` from `voice_diagnostics.py` by
   name, and offline-first is not optional in this stack.

   There is NO cloud path in this file now. It takes a `SynthesisBackend` —
   whatever local engine you already have — and refuses to run without one.
   It does not fall back to anything networked.

2. THE "REGION" PROFILES WERE gtts TLDs.

       {"code": "com", "name": "US English"}
       {"code": "co.uk", "name": "UK English"}

   Those are Google domain suffixes. They mean nothing to a local engine.
   Profiles are now free-form parameter dicts the backend interprets, so the
   diagnostic works against XTTS, GPT-SoVITS, or anything else without
   pretending a TLD is a voice.

3. MD5 FOR CONTENT COMPARISON.

   Not a security hole here — nobody is attacking a diagnostic — but MD5
   collides, and this file's entire purpose is deciding whether two outputs are
   identical. SHA-256 costs nothing at these sizes.

4. FAILURES KILLED THE WHOLE RUN.

   No try/except around synthesis. One failing profile aborted the report and
   left a partial directory. Each configuration is now independent, and
   failures appear in the report as failures.

5. `print()` for the completion notice, in a library.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

HASH_CHUNK = 8192


class SynthesisUnavailable(RuntimeError):
    """Raised when no local synthesis backend is attached."""


@runtime_checkable
class SynthesisBackend(Protocol):
    """
    A LOCAL speech synthesizer.

    Implementations must not make network calls. There is deliberately no
    interface here for a remote service and none should be added — this file
    previously shipped a user's text to a third party fourteen times per run.
    """

    name: str

    def synthesize(
        self, text: str, output_path: Path, **params: Any
    ) -> None:
        """Write audio for `text` to `output_path`. Raise on failure."""
        ...


#: Voice configurations to compare. Free-form parameter dicts, passed to the
#: backend as keyword arguments. The predecessor hardcoded gtts TLDs here.
DEFAULT_PROFILES: List[Dict[str, Any]] = [
    {"label": "default", "params": {}},
    {"label": "slow", "params": {"speed": 0.8}},
    {"label": "fast", "params": {"speed": 1.2}},
]


@dataclass
class ConfigurationResult:
    """One synthesis attempt."""

    label: str
    params: Dict[str, Any]
    filename: Optional[str] = None
    file_size: Optional[int] = None
    sha256: Optional[str] = None
    ok: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosticReport:
    """Outcome of a diagnostic run."""

    backend: str
    text: str
    generated_at: str
    output_directory: str
    results: List[ConfigurationResult] = field(default_factory=list)

    @property
    def succeeded(self) -> List[ConfigurationResult]:
        return [r for r in self.results if r.ok]

    @property
    def failed(self) -> List[ConfigurationResult]:
        return [r for r in self.results if not r.ok]

    @property
    def duplicate_groups(self) -> Dict[str, List[str]]:
        """
        Configurations whose audio is byte-identical.

        This is what the diagnostic is FOR: proving that a configuration
        labelled as a distinct voice actually produces distinct audio.
        """
        by_hash: Dict[str, List[str]] = {}
        for r in self.succeeded:
            if r.sha256:
                by_hash.setdefault(r.sha256, []).append(r.label)
        return {h: labels for h, labels in by_hash.items() if len(labels) > 1}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "text": self.text,
            "generated_at": self.generated_at,
            "output_directory": self.output_directory,
            "configurations_attempted": len(self.results),
            "configurations_succeeded": len(self.succeeded),
            "configurations_failed": len(self.failed),
            "unique_outputs": len({r.sha256 for r in self.succeeded if r.sha256}),
            "duplicate_groups": self.duplicate_groups,
            "all_distinct": not self.duplicate_groups,
            "results": [r.to_dict() for r in self.results],
        }

    def to_text(self) -> str:
        lines = [
            "Voice System Diagnostic Report",
            "=" * 32, "",
            f"Backend:   {self.backend}",
            f"Date:      {self.generated_at}",
            f"Test text: {self.text!r}", "",
            "Configurations", "-" * 14,
        ]
        for r in self.results:
            if r.ok:
                lines += [
                    f"{r.label}  params={r.params}",
                    f"  file:   {r.filename}",
                    f"  size:   {r.file_size} bytes",
                    f"  sha256: {r.sha256}", "",
                ]
            else:
                lines += [f"{r.label}  FAILED: {r.error}", ""]

        lines += [
            "Uniqueness", "-" * 10,
            f"Attempted: {len(self.results)}",
            f"Succeeded: {len(self.succeeded)}",
            f"Failed:    {len(self.failed)}",
            f"Unique outputs: {len({r.sha256 for r in self.succeeded if r.sha256})}",
            "",
        ]
        if self.duplicate_groups:
            lines += ["IDENTICAL OUTPUT GROUPS", "-" * 23,
                      "These configurations produced byte-identical audio. They",
                      "are not distinct voices.", ""]
            for i, (h, labels) in enumerate(self.duplicate_groups.items(), 1):
                lines += [f"Group {i}: {h[:16]}…"] + [f"  - {l}" for l in labels] + [""]
        elif self.succeeded:
            lines += ["Every successful configuration produced distinct audio.", ""]
        return "\n".join(lines)


def calculate_file_hash(file_path: Path | str) -> str:
    """SHA-256 of a file. Was MD5, in a file whose job is identity comparison."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as handle:
        while chunk := handle.read(HASH_CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_voice_diagnostic_report(
    backend: Optional[SynthesisBackend] = None,
    text: str = "This is a test of the voice system",
    profiles: Optional[List[Dict[str, Any]]] = None,
    output_directory: Optional[str] = None,
    extension: str = "wav",
) -> DiagnosticReport:
    """
    Run a diagnostic across voice configurations.

    Args:
        backend: A LOCAL synthesizer. Required — there is no cloud fallback.

    Raises:
        SynthesisUnavailable: when no backend is supplied. The predecessor
            reached for gtts here and sent the text off the machine.
    """
    if backend is None:
        raise SynthesisUnavailable(
            "voice_diagnostics requires a local SynthesisBackend. There is no "
            "cloud fallback: this module previously used gtts and sent the "
            "test text to Google once per configuration."
        )

    profiles = profiles or DEFAULT_PROFILES
    stamp = datetime.now(timezone.utc)
    directory = Path(
        output_directory or f"voice_test_{stamp.strftime('%Y%m%d_%H%M%S')}"
    )
    directory.mkdir(parents=True, exist_ok=True)

    report = DiagnosticReport(
        backend=getattr(backend, "name", type(backend).__name__),
        text=text,
        generated_at=stamp.isoformat(),
        output_directory=str(directory),
    )

    for profile in profiles:
        label = str(profile.get("label", "unnamed"))
        params = dict(profile.get("params") or {})
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
        filename = f"{safe}.{extension}"
        path = directory / filename
        result = ConfigurationResult(label=label, params=params, filename=filename)

        try:
            backend.synthesize(text, path, **params)
            if not path.exists():
                raise FileNotFoundError(f"backend reported success but {path} is absent")
            result.file_size = path.stat().st_size
            if result.file_size == 0:
                raise ValueError("backend wrote a zero-byte file")
            result.sha256 = calculate_file_hash(path)
            result.ok = True
        except Exception as exc:
            # One failing configuration must not abort the run. The predecessor
            # had no handler, so a single failure killed the whole report.
            logger.error("Configuration %r failed: %s", label, exc)
            result.error = str(exc)

        report.results.append(result)

    (directory / "voice_report.txt").write_text(report.to_text(), encoding="utf-8")
    (directory / "voice_report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )

    logger.info(
        "Diagnostic complete: %d/%d succeeded, %d duplicate group(s). Report: %s",
        len(report.succeeded), len(report.results),
        len(report.duplicate_groups), directory / "voice_report.txt",
    )
    return report


__all__ = [
    "SynthesisBackend", "SynthesisUnavailable", "ConfigurationResult",
    "DiagnosticReport", "generate_voice_diagnostic_report",
    "calculate_file_hash", "DEFAULT_PROFILES",
]

# ==============================================================================
# Patent Pending
# Christman-AI Family
# Shared-neutral implementation for internal system use.
# ==============================================================================
