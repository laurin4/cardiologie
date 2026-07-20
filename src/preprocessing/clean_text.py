"""
Generic clinical report text cleaning.

Conservative by design: normalizes whitespace and strips control characters but
preserves the original wording, casing, and clinical abbreviations so that
downstream rule-based evidence extraction can match verbatim.
"""

from __future__ import annotations

import re
import unicodedata

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_report_text(text: str, *, normalize_unicode: bool = True) -> str:
    """
    Return a cleaned copy of *text*.

    - Optionally applies Unicode NFC normalization.
    - Removes non-printable control characters (keeps tabs/newlines meaning).
    - Collapses runs of spaces/tabs and excessive blank lines.
    - Trims leading/trailing whitespace.
    """
    if not text:
        return ""
    s = str(text)
    if normalize_unicode:
        s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _CONTROL_CHARS_RE.sub(" ", s)
    s = _MULTI_SPACE_RE.sub(" ", s)
    s = _MULTI_NEWLINE_RE.sub("\n\n", s)
    # Strip trailing spaces on each line.
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    return s.strip()
