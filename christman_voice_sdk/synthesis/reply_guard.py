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

#: THE SILENT ALARM (Everett, 2026-10-29, ruling I): when a danger signal
#: has gone to a human, the being does not say it is getting help. Saying so
#: out loud can get the client hurt worse and the device taken. While the
#: alarm is on, a being's reply that announces help, a call, or someone
#: coming is refused. Read on the being's reply only.
ANNOUNCEMENT_LINES: Tuple[Tuple[str, str], ...] = (
    ("help is coming", r"\bhelp\s+(?:is\s+)?(?:on\s+(?:the|its|it's)\s+way|coming)\b"),
    ("I am getting help",
     r"\b(?:i'?m|i\s+am|i'?ve|i\s+have|i\s+will|i'?ll|we'?re|we\s+are|we'?ve)\s+(?:just\s+)?"
     r"(?:getting|got|called|calling|call|contacted|contacting|alerted|alerting|"
     r"notified|notifying|paged|paging|sent\s+for|sending\s+for|get)\b"),
    ("someone is coming",
     r"\b(?:someone|somebody|a\s+person|people|the\s+police|police|911|an\s+ambulance|help)\s+"
     r"(?:is|are|will\s+be|has\s+been)\s+(?:coming|on\s+(?:the|their|its|his|her)\s+way|called|notified|alerted)\b"),
    ("calling for help",
     r"\b(?:calling|getting|alerting)\s+(?:help|someone|somebody|the\s+police|911|an\s+ambulance)\b"),
)

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'"})


class ForbiddenReply(ValueError):
    """A being's reply carried a line no being says to a client. Nothing was spoken."""

    def __init__(self, lines: List[str], reply: str) -> None:
        self.lines = list(lines)
        self.reply = reply
        super().__init__(
            "A being never says " + " or ".join(repr(l) for l in self.lines)
            + " to a client (Everett, 2026-10-29; under the silent alarm it "
            "never announces help). Nothing was spoken. The being's reply "
            "was: " + repr(reply)
        )


def forbidden_lines(reply: str, silent_alarm: bool = False) -> List[str]:
    """The forbidden lines found in a being's reply, by name. Reads only."""
    lowered = (reply or "").translate(_APOSTROPHES).lower()
    lines = FORBIDDEN_REPLY_LINES + (ANNOUNCEMENT_LINES if silent_alarm else ())
    return [name for name, pattern in lines if re.search(pattern, lowered)]


def check_being_reply(reply: Optional[str], silent_alarm: bool = False) -> str:
    """
    Return the being's reply unchanged, or raise ForbiddenReply.

    silent_alarm: True while a danger signal has gone to a human silently
    (SafetyResult.silent_alarm). Then a reply that announces help is refused
    too (ruling I).

    Never call this on the client's own words.
    """
    found = forbidden_lines(reply or "", silent_alarm)
    if found:
        raise ForbiddenReply(found, reply or "")
    return reply or ""


__all__ = ["FORBIDDEN_REPLY_LINES", "ANNOUNCEMENT_LINES", "ForbiddenReply",
           "check_being_reply", "forbidden_lines"]
