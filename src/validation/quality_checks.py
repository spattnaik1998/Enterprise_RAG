"""
Document Quality Checks
------------------------
Individual, composable check functions.  Each returns:
    (passed: bool, message: str)

The validator calls these and aggregates results into a ValidationResult.
"""
from __future__ import annotations

import re


# --- Individual Checks --------------------------------------------------------

def check_min_length(content: str, min_chars: int = 150) -> tuple[bool, str]:
    n = len(content)
    if n >= min_chars:
        return True, f"Length OK ({n:,} chars)"
    return False, f"Too short: {n} < {min_chars} chars"


def check_max_length(content: str, max_chars: int = 500_000) -> tuple[bool, str]:
    n = len(content)
    if n <= max_chars:
        return True, "Length within limit"
    return False, f"Too long: {n:,} > {max_chars:,} chars"


def check_language(content: str, allowed: list[str] | None = None) -> tuple[bool, str]:
    """Detect language using langdetect (best-effort; accepts on uncertainty)."""
    if allowed is None:
        allowed = ["en"]
    try:
        from langdetect import LangDetectException, detect

        try:
            lang = detect(content[:1_500])
            if lang in allowed:
                return True, f"Language: {lang}"
            return False, f"Language '{lang}' not in allowed {allowed}"
        except LangDetectException:
            return True, "Language detection inconclusive (accepted)"
    except ImportError:
        return True, "langdetect not installed - language check skipped"


def check_alphabetic_ratio(content: str, min_ratio: float = 0.30) -> tuple[bool, float, str]:
    """Require at least `min_ratio` fraction of alphabetic characters."""
    total = len(content)
    if total == 0:
        return False, 0.0, "Empty content"
    ratio = sum(1 for c in content if c.isalpha()) / total
    if ratio >= min_ratio:
        return True, ratio, f"Alpha ratio OK ({ratio:.1%})"
    return False, ratio, f"Low alpha ratio: {ratio:.1%} < {min_ratio:.0%}"


def check_boilerplate(content: str) -> tuple[bool, str]:
    """Detect obvious low-quality / boilerplate patterns."""
    patterns = [
        r"lorem ipsum",
        r"this page (is|has been) intentionally left blank",
        r"^(\W|\s)+$",  # only non-word characters
    ]
    lower = content.lower()
    for pat in patterns:
        if re.search(pat, lower):
            return False, f"Boilerplate pattern detected: '{pat}'"
    return True, "No boilerplate detected"


def check_url_format(url: str | None) -> tuple[bool, str]:
    if url is None:
        return True, "URL field absent (optional)"
    if re.match(r"^https?://", url):
        return True, "URL format valid"
    return False, f"Invalid URL format: {url[:80]}"


def check_duplicate(checksum: str, seen: set[str]) -> tuple[bool, str]:
    if checksum in seen:
        return False, f"Exact duplicate (checksum prefix: {checksum[:12]})"
    seen.add(checksum)
    return True, "No duplicate"


# --- Composite Quality Score --------------------------------------------------

def compute_quality_score(content: str, title: str) -> float:
    """
    Heuristic quality score in [0.0, 1.0].

    Weights:
        0.30  content length
        0.20  title quality
        0.20  sentence length distribution
        0.20  alphabetic character ratio
        0.10  lexical diversity (unique-word ratio)
    """
    score = 0.0

    # - Length -
    n = len(content)
    if n >= 800:
        score += 0.30
    elif n >= 300:
        score += 0.20
    elif n >= 150:
        score += 0.10

    # - Title -
    if title and 5 <= len(title) <= 300:
        score += 0.20
    elif title:
        score += 0.08

    # - Sentence length distribution -
    sentences = re.split(r"[.!?]+", content)
    avg_words = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    if 6 <= avg_words <= 45:
        score += 0.20
    elif avg_words > 0:
        score += 0.08

    # - Alphabetic ratio -
    alpha = sum(1 for c in content if c.isalpha()) / max(n, 1)
    if alpha >= 0.55:
        score += 0.20
    elif alpha >= 0.30:
        score += 0.10

    # - Lexical diversity -
    words = content.lower().split()
    if words:
        diversity = len(set(words)) / len(words)
        if diversity >= 0.35:
            score += 0.10
        elif diversity >= 0.20:
            score += 0.05

    return round(min(score, 1.0), 4)
