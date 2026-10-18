# ==============================================================================
# © 2025 Everett Nathaniel Christman & Misty Gail Christman
# The Christman AI Project — Luma Cognify AI
# Truth. Dignity. Protection. Transparency. No Erasure.
# ==============================================================================

"""
christman_sound.lexicon — words to phones, from Everett's own Vosk ear.

WHAT CHANGED AND WHY (added 2026-10-17)
---------------------------------------
The sovereign synthesizer (christman_synth.c, the C port of Codec War) speaks
Codec War's 40 phones. Codec War's own word list is about a hundred words and
spells every other word letter by letter (phonemes.ts:208-218), which is a
guess presented as a pronunciation. The ear Everett already runs — Vosk with
vosk-model-en-us-0.22 — carries a pronunciation lexicon of about 484,651
lines in graph/phones/align_lexicon.txt. This module reads that file and maps
its phones onto Codec War's.

    christman christman k_B r_I I_I s_I t_I m_I V_I n_E
    hello hello h_B @_I l_I oU_E

Each line is: the word, the word again, then its phones with Kaldi's word-
position markers (_B begin, _I inside, _E end, _S a one-phone word). The
markers are stripped; the phone is mapped through VOSK_TO_CODECWAR below.

Rules this module keeps, every one a refusal and never a substitute:
  * No lexicon path (argument or $CHRISTMAN_LEXICON_PATH) -> LexiconUnavailable.
    There is no fallback to letter-by-letter spelling.
  * A word not in the lexicon -> WordNotInLexicon, naming every missing word.
  * A lexicon phone with no Codec War counterpart is listed in NOT_MAPPED and
    raises PhoneNotMapped when a word that uses it is asked for. It is never
    dropped and never replaced by a near neighbour. Since 2026-10-18 only the
    ten non-speech models are left there (see the ruling below).
  * A line that is not in the align_lexicon format -> LexiconFormatError with
    the line number. The file is not half-loaded.
The lexicon is never vendored into this repository and never downloaded by
this code. It lives on the machine with the Vosk model.

WHERE THE PHONE SET CAME FROM
-----------------------------
Vosk side: graph/phones.txt of vosk-model-en-us-0.22-lgraph, which is built
on the same acoustic model as vosk-model-en-us-0.22 and so has the same phone
inventory: 54 phones once the position markers and Kaldi's <eps> and #n
disambiguation symbols are set aside. 44 of them map onto Codec War's phones
(every one of Codec War's 40 is reached); the 10 non-speech models are
NOT_MAPPED. The speech phones are X-SAMPA. Only that
one 2 KB file was read (by an HTTP range read of the model archive); the
lexicon itself was not downloaded. To confirm against the lexicon on the Mac:

    cut -d' ' -f3- align_lexicon.txt | tr ' ' '\\n' | sed 's/_[BIES]$//' | sort | uniq -c

Codec War side: PHONES in phonemes.ts:17-58 (ARPAbet), mirrored in the C
table in christman_synth.c, which is the one copy the synthesizer reads.

EVERETT'S RULING, 2026-10-18 — THE FLAP AND THE LONG E
------------------------------------------------------
Everett ruled that Codec War can already make both sounds from what it has,
so neither may refuse. Each mapping row now carries a duration factor, and
two rows use one other than 1.0, shown in the table with their reasons:
    4   -> D  x 0.5     the flap is a shortened D
    E:  -> EH x 1.625   the long E is EH held long
The factor multiplies the phone's Codec War table duration, on top of Codec
War's own word-position factor (1.05 at the edges of a word, 0.95 inside).
These are chosen starting values with their reasons written beside them,
held until measurements of Everett's own speech replace them.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

__all__ = [
    "LexiconError",
    "LexiconUnavailable",
    "LexiconFormatError",
    "WordNotInLexicon",
    "PhoneNotMapped",
    "VOSK_TO_CODECWAR",
    "DURATION_REASONS",
    "NOT_MAPPED",
    "VOSK_PHONES",
    "LEXICON_ENV",
    "Lexicon",
    "load_lexicon",
    "normalize_word",
    "strip_position",
    "map_phone",
]

LEXICON_ENV = "CHRISTMAN_LEXICON_PATH"


class LexiconError(RuntimeError):
    """Base for every lexicon failure. Nothing was looked up."""


class LexiconUnavailable(LexiconError):
    """No lexicon file is configured, or the configured file cannot be read."""


class LexiconFormatError(LexiconError):
    """A line in the file is not in the align_lexicon format."""


class WordNotInLexicon(LexiconError, KeyError):
    """One or more words have no pronunciation in the lexicon."""

    def __init__(self, words: List[str], path: Optional[Path] = None) -> None:
        self.words = list(words)
        self.path = path
        where = f" ({path})" if path else ""
        super().__init__(f"not in the lexicon{where}: {', '.join(repr(w) for w in self.words)}")

    def __str__(self) -> str:  # KeyError would otherwise quote the whole message
        return self.args[0]


class PhoneNotMapped(LexiconError):
    """A lexicon phone has no Codec War counterpart. The word cannot be voiced faithfully."""

    def __init__(self, phone: str, word: Optional[str] = None) -> None:
        self.phone = phone
        self.word = word
        if phone in NOT_MAPPED:
            why = f"listed in NOT_MAPPED: {NOT_MAPPED[phone]}"
        else:
            why = "not in the known vosk-model-en-us-0.22 phone inventory"
        in_word = f" in {word!r}" if word else ""
        super().__init__(f"lexicon phone {phone!r}{in_word} has no Codec War phone ({why})")


# ------------------------------------------------------------------------------
# THE MAPPING. Left: vosk-model-en-us-0.22 phone (X-SAMPA). Right: (Codec War
# phone id, duration factor). The id is ARPAbet with its phonemes.ts line in
# the comment; the factor multiplies that phone's Codec War table duration and
# is 1.0 on every row except the two Everett ruled on (2026-10-18), whose
# reasons are in DURATION_REASONS just below. Built row by row from both
# inventories. Two Vosk vowels land on each of AH and ER because Codec War,
# like ARPAbet without stress digits, has one phone where X-SAMPA has two: the
# stressed and unstressed forms. Codec War's own word list already writes
# schwa as AH ("filament" F IH L AH M AH N T, phonemes.ts:68).
# ------------------------------------------------------------------------------
VOSK_TO_CODECWAR: Dict[str, Tuple[str, float]] = {
    # vowels
    "A": ("AA", 1.0),    # ɑ  father             phonemes.ts:19
    "{": ("AE", 1.0),    # æ  cat                phonemes.ts:20
    "V": ("AH", 1.0),    # ʌ  strut (stressed)   phonemes.ts:21
    "@": ("AH", 1.0),    # ə  schwa (unstressed) phonemes.ts:21
    "O": ("AO", 1.0),    # ɔ  thought            phonemes.ts:22
    "aU": ("AW", 1.0),   # aʊ mouth              phonemes.ts:23
    "aI": ("AY", 1.0),   # aɪ price              phonemes.ts:24
    "E": ("EH", 1.0),    # ɛ  dress              phonemes.ts:25
    "E:": ("EH", 1.625),  # ɛː long E: EH held long — see DURATION_REASONS["E:"]
    "3`": ("ER", 1.0),   # ɝ  nurse (stressed)   phonemes.ts:26
    "@`": ("ER", 1.0),   # ɚ  letter (unstressed) phonemes.ts:26
    "eI": ("EY", 1.0),   # eɪ face               phonemes.ts:27
    "I": ("IH", 1.0),    # ɪ  kit                phonemes.ts:28
    "i": ("IY", 1.0),    # i  fleece             phonemes.ts:29
    "oU": ("OW", 1.0),   # oʊ goat               phonemes.ts:30
    "OI": ("OY", 1.0),   # ɔɪ choice             phonemes.ts:31
    "U": ("UH", 1.0),    # ʊ  foot               phonemes.ts:32
    "u": ("UW", 1.0),    # u  goose              phonemes.ts:33
    # stops
    "b": ("B", 1.0),     # phonemes.ts:34
    "d": ("D", 1.0),     # phonemes.ts:35
    "g": ("G", 1.0),     # phonemes.ts:36
    "p": ("P", 1.0),     # phonemes.ts:37
    "t": ("T", 1.0),     # phonemes.ts:38
    "4": ("D", 0.5),     # ɾ  flap (the t in "butter"): a shortened D — see DURATION_REASONS["4"]
    "k": ("K", 1.0),     # phonemes.ts:39
    # fricatives
    "f": ("F", 1.0),     # phonemes.ts:40
    "v": ("V", 1.0),     # phonemes.ts:41
    "T": ("TH", 1.0),    # θ thin   phonemes.ts:42
    "D": ("DH", 1.0),    # ð this   phonemes.ts:43
    "s": ("S", 1.0),     # phonemes.ts:44
    "z": ("Z", 1.0),     # phonemes.ts:45
    "S": ("SH", 1.0),    # ʃ        phonemes.ts:46
    "Z": ("ZH", 1.0),    # ʒ        phonemes.ts:47
    "h": ("HH", 1.0),    # phonemes.ts:48
    # affricates
    "tS": ("CH", 1.0),   # tʃ       phonemes.ts:49
    "dZ": ("JH", 1.0),   # dʒ       phonemes.ts:50
    # nasals
    "m": ("M", 1.0),     # phonemes.ts:51
    "n": ("N", 1.0),     # phonemes.ts:52
    "N": ("NG", 1.0),    # ŋ        phonemes.ts:53
    # liquids, glides
    "l": ("L", 1.0),     # phonemes.ts:54
    "r": ("R", 1.0),     # phonemes.ts:55
    "w": ("W", 1.0),     # phonemes.ts:56
    "j": ("Y", 1.0),     # j yes    phonemes.ts:57
    # silence is silence on both sides
    "SIL": ("SIL", 1.0),  # phonemes.ts:18
}

#: Why the two rows that are not x 1.0 have the factor they have.
DURATION_REASONS: Dict[str, str] = {
    "4": (
        "x 0.5. A flap is a single brief tap of the tongue, not a closure "
        "held and released. Codec War's D is already its shortest phone, "
        "0.05 s (phonemes.ts:35, tied with T); half of it, 0.025 s, keeps "
        "D's own shape — the near-silent closure then the burst, which "
        "synth.ts:51-52 lays out over whatever duration the phone is given "
        "— compressed into a single tap. A chosen starting value, held until "
        "Everett's own flaps are measured."
    ),
    "E:": (
        "x 1.625. EH is 0.08 s in Codec War (phonemes.ts:25). Codec War's "
        "own long vowels, EY and OW, are 0.13 s (phonemes.ts:27, :30); "
        "0.08 x 1.625 = 0.13 s, so the long E is held exactly as long as "
        "Codec War already holds its long vowels. Same formants as EH, so "
        "the same vowel, held long. A chosen starting value, held until "
        "Everett's own speech is measured."
    ),
}

# ------------------------------------------------------------------------------
# NOT MAPPED. Every lexicon phone with no Codec War counterpart: since
# Everett's 2026-10-18 ruling, only the ten non-speech models. They are not
# sounds of speech, so there is nothing for Codec War to make. Hitting one
# raises PhoneNotMapped.
# ------------------------------------------------------------------------------
NOT_MAPPED: Dict[str, str] = {
    "BRH": "breath (non-speech model)",
    "CGH": "cough (non-speech model)",
    "LAUGHTER": "laughter (non-speech model)",
    "NOISE": "noise (non-speech model)",
    "NSN": "non-speech noise (non-speech model)",
    "OOV": "out-of-vocabulary placeholder (non-speech model)",
    "SMK": "lip smack (non-speech model)",
    "SPN": "spoken noise (non-speech model)",
    "UHH": "hesitation 'uh' (filler model, not a phone sequence)",
    "UM": "hesitation 'um' (filler model, not a phone sequence)",
}

#: The full vosk-model-en-us-0.22 inventory this table was built from.
VOSK_PHONES = frozenset(VOSK_TO_CODECWAR) | frozenset(NOT_MAPPED)

_POSITION = re.compile(r"_[BIES]$")


def strip_position(phone: str) -> str:
    """Remove Kaldi's word-position marker: 'k_B' -> 'k', 'aI_S' -> 'aI', 'SIL' -> 'SIL'."""
    return _POSITION.sub("", phone)


def map_phone(vosk_phone: str, word: Optional[str] = None) -> Tuple[str, float]:
    """
    One Vosk phone (marker already stripped) to (Codec War id, duration
    factor), or PhoneNotMapped.
    """
    try:
        return VOSK_TO_CODECWAR[vosk_phone]
    except KeyError:
        raise PhoneNotMapped(vosk_phone, word) from None


_NON_WORD_1 = re.compile(r"[^a-zA-Z'-]")
_NON_WORD_2 = re.compile(r"[^a-z']")


def normalize_word(raw: str) -> str:
    """
    The lookup key for a typed word, by Codec War's own rule: strip to
    letters, apostrophes and hyphens (phonemes.ts:231), then lower-case and
    keep letters and apostrophes (phonemes.ts:209). "Hello," -> "hello",
    "don't" -> "don't". May return "" for a token with no letters.
    """
    return _NON_WORD_2.sub("", _NON_WORD_1.sub("", raw).lower())


class Lexicon:
    """
    An align_lexicon.txt loaded into memory. Pronunciations keep file order.
    Mapping to Codec War happens on lookup, so a file that contains unmapped
    phones still loads; the failure comes when a word that needs one is asked for.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self._entries: Dict[str, List[Tuple[str, ...]]] = {}
        self._load()

    def _load(self) -> None:
        # Read as bytes and decode line by line (2026-10-18). Opened in text
        # mode, a byte that is not UTF-8 raised a bare UnicodeDecodeError out
        # of the loop — not a LexiconError — so synthesis_readiness and the
        # mouth's constructor crashed instead of reporting why, and the line
        # could not be named. Now it is LexiconFormatError with the line.
        try:
            handle = self.path.open("rb")
        except OSError as exc:
            raise LexiconUnavailable(f"cannot read lexicon {self.path}: {exc}") from exc
        intern: Dict[str, str] = {}
        with handle:
            for lineno, raw in enumerate(handle, 1):
                try:
                    line = raw.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise LexiconFormatError(
                        f"{self.path}:{lineno}: not UTF-8 ({exc.reason} at byte {exc.start})"
                    ) from exc
                fields = line.split()
                if not fields:
                    continue  # a blank line carries no entry
                if len(fields) < 3:
                    raise LexiconFormatError(
                        f"{self.path}:{lineno}: expected 'word word phone...', got {line.rstrip()!r}"
                    )
                if fields[0] != fields[1]:
                    raise LexiconFormatError(
                        f"{self.path}:{lineno}: the first two fields must be the same word "
                        f"(align_lexicon format), got {fields[0]!r} and {fields[1]!r}"
                    )
                phones = tuple(intern.setdefault(p, p) for p in (strip_position(f) for f in fields[2:]))
                self._entries.setdefault(fields[0], []).append(phones)
        if not self._entries:
            raise LexiconFormatError(f"{self.path}: no entries")

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, word: str) -> bool:
        return word in self._entries

    def pronunciations(self, word: str) -> List[Tuple[str, ...]]:
        """Every pronunciation of `word` as Vosk phones (markers stripped), in file order."""
        try:
            return list(self._entries[word])
        except KeyError:
            raise WordNotInLexicon([word], self.path) from None

    def phones(self, word: str) -> List[str]:
        """
        Codec War phone ids for `word` (already a lookup key; see
        normalize_word). Uses the FIRST pronunciation in file order — the
        file's own order is the only ranking it carries. Raises
        WordNotInLexicon or PhoneNotMapped.
        """
        return [pid for pid, _ in self.phones_timed(word)]

    def phones_timed(self, word: str) -> List[Tuple[str, float]]:
        """
        (Codec War id, duration factor) for each phone of `word`, first
        pronunciation in file order. The factor is the mapping table's
        (1.0 except the flap and the long E). Raises WordNotInLexicon or
        PhoneNotMapped.
        """
        first = self.pronunciations(word)[0]
        return [map_phone(p, word) for p in first]


_cache: Dict[Tuple[str, int, int], Lexicon] = {}
_cache_lock = threading.Lock()


def load_lexicon(path: Optional[Union[str, Path]] = None) -> Lexicon:
    """
    The lexicon at `path`, else at $CHRISTMAN_LEXICON_PATH. Loaded once per
    file version (path, size, modification time) and reused. With neither set,
    raises LexiconUnavailable — there is no fallback spelling.
    """
    chosen = path if path is not None else os.environ.get(LEXICON_ENV)
    if not chosen:
        raise LexiconUnavailable(
            f"no lexicon configured: pass a path or set ${LEXICON_ENV} to "
            "vosk-model-en-us-0.22/graph/phones/align_lexicon.txt"
        )
    p = Path(chosen).expanduser()
    try:
        st = p.stat()
    except OSError as exc:
        raise LexiconUnavailable(f"cannot read lexicon {p}: {exc}") from exc
    if not p.is_file():
        raise LexiconUnavailable(f"lexicon path is not a file: {p}")
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    with _cache_lock:
        lex = _cache.get(key)
        if lex is None:
            lex = Lexicon(p)
            _cache[key] = lex
        return lex
