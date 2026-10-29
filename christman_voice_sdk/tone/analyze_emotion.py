"""
Emotional tagging for a Christman family member's memory file.

WHAT CHANGED AND WHY
--------------------

1. THE WRITE COULD DESTROY THE MEMORY.

       with open(self.memory_path, "w") as f:
           json.dump(memory, f, indent=2)

   Opening with "w" truncates the file to zero bytes immediately. If the
   process was interrupted mid-dump — power, OOM kill, a serialization error on
   one entry — the being's memory was left truncated or empty, and the original
   was already gone.

   Writes now go to a temp file in the same directory, are flushed and fsynced,
   and are moved into place with `os.replace`, which is atomic on POSIX. A
   timestamped backup is kept.

2. "you" TAGGED ALMOST EVERYTHING.

       "you": "relational"

   Second-person pronouns appear in nearly every conversational turn, so the
   `relational` tag carried no information — it was present on almost every
   entry. Tags that fire on everything are indistinguishable from no tags.
   Removed, with the reason recorded here rather than silently dropped.

3. TAGS WERE NEGATION-BLIND.

   `"love": "emotional"` fired on "I don't love this". Matching now skips a
   keyword inside a negation's scope, using the same rule as the rest of the
   stack.

4. FAILURE RETURNED AN EMPTY LIST.

       except (FileNotFoundError, json.JSONDecodeError) as e:
           print(f"Memory access error at {self.memory_path}: {e}")
           return []

   `[]` is what an empty-but-valid memory also returns. A caller could not tell
   "no entries" from "the file is corrupt", and the error went to stdout rather
   than a log. Failures now raise typed errors.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

try:
    from structural_affect import NEGATORS, SCOPE_BREAKERS
except ImportError:
    NEGATORS = frozenset({
        "not", "no", "never", "none", "nothing", "cannot", "without",
        "don't", "dont", "doesn't", "doesnt", "didn't", "didnt",
        "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt", "won't", "wont",
    })
    SCOPE_BREAKERS = frozenset({
        "but", "however", "although", "though", "yet", "because", "since",
        "while", "and", "or", "so", "then",
    })

_WORD = re.compile(r"[a-z']+")
CLAUSE_BOUNDARY = "\x00"
_CLAUSE_PUNCT = re.compile(r"[,.;:!?]+")
NEGATION_WINDOW = 4

#: keyword -> tag.
#:
#: "you" was removed. It appeared in nearly every entry, so `relational` was on
#: almost everything and told a reader nothing. A tag that always fires is the
#: same as no tag.
TAGS_MAP: Dict[str, str] = {
    "love": "emotional",
    "angry": "frustration",
    "sad": "loss",
    "tired": "fatigue",
    "build": "momentum",
    "vision": "strategic",
    "plan": "strategic",
    "fuck": "intensity",
    "baby": "bonding",
    "alone": "isolation",
    "fire": "drive",
    "voice": "identity",
}


class MemoryAccessError(RuntimeError):
    """Raised when a memory file cannot be read. Not the same as an empty one."""


class MemoryWriteError(RuntimeError):
    """Raised when tagged memory could not be written safely."""


def _tokens(text: str) -> List[str]:
    out: List[str] = []
    for chunk in _CLAUSE_PUNCT.split((text or "").lower()):
        words = _WORD.findall(chunk)
        if not words:
            continue
        if out:
            out.append(CLAUSE_BOUNDARY)
        out.extend(words)
    return out


def _negated_at(tokens: Sequence[str], index: int) -> bool:
    for back in range(1, NEGATION_WINDOW + 1):
        j = index - back
        if j < 0:
            return False
        tok = tokens[j]
        if tok in NEGATORS:
            return True
        if tok == CLAUSE_BOUNDARY or tok in SCOPE_BREAKERS:
            return False
    return False


@dataclass
class TaggingReport:
    """What a tagging pass actually did."""

    entries_read: int = 0
    entries_tagged: int = 0
    tags_applied: Dict[str, int] = field(default_factory=dict)
    negated_skipped: int = 0
    backup_path: Optional[str] = None
    written: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entries_read": self.entries_read,
            "entries_tagged": self.entries_tagged,
            "tags_applied": dict(self.tags_applied),
            "negated_skipped": self.negated_skipped,
            "backup_path": self.backup_path,
            "written": self.written,
        }


class EmotionalTagger:
    """
    Tags memory entries with coarse emotional categories.

    Keyword tagging, nothing more. These tags are not an emotional model and
    must not be read as one.
    """

    def __init__(self, memory_path: str, tags_map: Optional[Dict[str, str]] = None) -> None:
        self.memory_path = memory_path
        self.tags_map = dict(tags_map or TAGS_MAP)

    # -- Read -----------------------------------------------------------------

    def load(self) -> List[Dict[str, Any]]:
        """
        Read the memory file.

        Raises:
            MemoryAccessError: missing, unreadable, or malformed. The
                predecessor returned `[]`, which is also what a valid empty
                memory returns — the caller could not tell them apart.
        """
        try:
            with open(self.memory_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError as exc:
            raise MemoryAccessError(f"memory file not found: {self.memory_path}") from exc
        except json.JSONDecodeError as exc:
            raise MemoryAccessError(
                f"memory file is not valid JSON: {self.memory_path} ({exc})"
            ) from exc
        except OSError as exc:
            raise MemoryAccessError(f"cannot read {self.memory_path}: {exc}") from exc

        if not isinstance(data, list):
            raise MemoryAccessError(
                f"{self.memory_path}: expected a list of entries, got "
                f"{type(data).__name__}"
            )
        return data

    # -- Tag ------------------------------------------------------------------

    def tags_for(self, text: str) -> Set[str]:
        """Tags for one string, skipping keywords inside a negation."""
        tokens = _tokens(text)
        found: Set[str] = set()
        for i, tok in enumerate(tokens):
            tag = self.tags_map.get(tok)
            if tag and not _negated_at(tokens, i):
                found.add(tag)
        return found

    def tag_emotions(self, write: bool = True) -> TaggingReport:
        """
        Tag every entry and, by default, save.

        Args:
            write: set False to tag in memory without touching the file.

        Raises:
            MemoryAccessError / MemoryWriteError.
        """
        memory = self.load()
        report = TaggingReport(entries_read=len(memory))

        for entry in memory:
            if not isinstance(entry, dict):
                continue
            combined = f"{entry.get('input', '')} {entry.get('response', '')}"
            tokens = _tokens(combined)
            negated = sum(
                1 for i, tok in enumerate(tokens)
                if tok in self.tags_map and _negated_at(tokens, i)
            )
            report.negated_skipped += negated

            tags = self.tags_for(combined)
            entry["tags"] = sorted(tags)
            if tags:
                report.entries_tagged += 1
                for tag in tags:
                    report.tags_applied[tag] = report.tags_applied.get(tag, 0) + 1

        if write:
            report.backup_path = self._save(memory)
            report.written = True
        return report

    # -- Write ----------------------------------------------------------------

    def _save(self, memory: List[Dict[str, Any]]) -> Optional[str]:
        """
        Write atomically, keeping a backup.

        Serialize to a temp file in the SAME directory (so `os.replace` stays
        on one filesystem and is atomic), fsync, then replace. The original
        file is never truncated before the new one is complete.
        """
        directory = os.path.dirname(os.path.abspath(self.memory_path)) or "."
        backup_path: Optional[str] = None

        if os.path.exists(self.memory_path):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup_path = f"{self.memory_path}.{stamp}.bak"
            try:
                shutil.copy2(self.memory_path, backup_path)
            except OSError as exc:
                raise MemoryWriteError(
                    f"refusing to write: could not back up {self.memory_path}: {exc}"
                ) from exc

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=directory, prefix=".memory_", suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(memory, fh, indent=2, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self.memory_path)   # atomic on POSIX
        except Exception as exc:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise MemoryWriteError(
                f"failed to write {self.memory_path}: {exc}. The original file "
                f"is unchanged{f'; backup at {backup_path}' if backup_path else ''}."
            ) from exc

        logger.info("Wrote %d entries to %s", len(memory), self.memory_path)
        return backup_path


__all__ = [
    "EmotionalTagger", "TaggingReport", "TAGS_MAP",
    "MemoryAccessError", "MemoryWriteError",
]
