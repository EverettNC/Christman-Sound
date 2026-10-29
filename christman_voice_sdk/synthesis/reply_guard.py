# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================
"""
Lines a being never says to a client.

Everett, 2026-10-29 (ruling E): "you're not okay" is a statement said at
someone, and as a hard rule no being ever says "you're not okay" or "calm
down" to a client.

This guard stands ONLY on the being's own reply: the path where the being
speaks for itself (`speech_response.speak`, and `generate_speech_audio(...,
being_reply=True)`). It is never run on the client's own words. A client who
types "calm down" to someone, or "you're not okay", is spoken exactly as
typed; nothing here reads, changes or holds the client's words.

When a being's reply carries one of these lines, nothing is spoken and
ForbiddenReply is raised with a plain reason. The reply is not edited into
something else: the being's caller decides what to say instead.

The list is fixed and short on purpose. Each line is matched as words, in
any case, with straight or curly apostrophes, wherever it sits in the reply.
"if you're not okay WITH that, we can stop" is a question about consent, not
a statement about the client, and is left alone, as "not okay with it" is
left alone everywhere else in this package.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

#: (name, pattern). Read on the being's reply only.
FORBIDDEN_REPLY_LINES: Tuple[Tuple[str, str], ...] = (
    ("you're not okay",
     r"\byou(?:'re|re|\s+are)\s+not\s+(?:really\s+|very\s+)?"
     r"(?:okay|ok|alright|all\s+right)\b(?!\s+with\b)"),
    ("calm down", r"\bcalm\s+(?:yourself\s+)?down\b"),
)

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'"})


class ForbiddenReply(ValueError):
    """A being's reply carried a line no being says to a client. Nothing was spoken."""

    def __init__(self, lines: List[str], reply: str) -> None:
        self.lines = list(lines)
        self.reply = reply
        super().__init__(
            "A being never says " + " or ".join(repr(l) for l in self.lines)
            + " to a client (Everett, 2026-10-29). Nothing was spoken. "
            "The being's reply was: " + repr(reply)
        )


def forbidden_lines(reply: str) -> List[str]:
    """The forbidden lines found in a being's reply, by name. Reads only."""
    lowered = (reply or "").translate(_APOSTROPHES).lower()
    return [name for name, pattern in FORBIDDEN_REPLY_LINES if re.search(pattern, lowered)]


def check_being_reply(reply: Optional[str]) -> str:
    """
    Return the being's reply unchanged, or raise ForbiddenReply.

    Never call this on the client's own words.
    """
    found = forbidden_lines(reply or "")
    if found:
        raise ForbiddenReply(found, reply or "")
    return reply or ""


__all__ = ["FORBIDDEN_REPLY_LINES", "ForbiddenReply", "check_being_reply", "forbidden_lines"]
