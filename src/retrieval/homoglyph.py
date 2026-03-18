"""
Homoglyph Normalizer
---------------------
Normalizes Unicode confusables (Cyrillic/Greek/math characters that look
like Latin letters) to prevent injection bypass via visual spoofing.

Example: Cyrillic 'a' (U+0430) -> Latin 'a' (U+0061)
"""
from __future__ import annotations

import unicodedata

# Common confusable mappings (Cyrillic, Greek, mathematical, fullwidth)
_CONFUSABLES: dict[str, str] = {
    # Cyrillic -> Latin
    "\u0430": "a",  # Cyrillic Small Letter A
    "\u0435": "e",  # Cyrillic Small Letter Ie
    "\u043e": "o",  # Cyrillic Small Letter O
    "\u0440": "p",  # Cyrillic Small Letter Er
    "\u0441": "c",  # Cyrillic Small Letter Es
    "\u0443": "y",  # Cyrillic Small Letter U
    "\u0445": "x",  # Cyrillic Small Letter Ha
    "\u0410": "A",  # Cyrillic Capital Letter A
    "\u0412": "B",  # Cyrillic Capital Letter Ve
    "\u0415": "E",  # Cyrillic Capital Letter Ie
    "\u041a": "K",  # Cyrillic Capital Letter Ka
    "\u041c": "M",  # Cyrillic Capital Letter Em
    "\u041d": "H",  # Cyrillic Capital Letter En
    "\u041e": "O",  # Cyrillic Capital Letter O
    "\u0420": "P",  # Cyrillic Capital Letter Er
    "\u0421": "C",  # Cyrillic Capital Letter Es
    "\u0422": "T",  # Cyrillic Capital Letter Te
    "\u0425": "X",  # Cyrillic Capital Letter Ha
    # Greek -> Latin
    "\u0391": "A",  # Greek Capital Alpha
    "\u0392": "B",  # Greek Capital Beta
    "\u0395": "E",  # Greek Capital Epsilon
    "\u0397": "H",  # Greek Capital Eta
    "\u0399": "I",  # Greek Capital Iota
    "\u039a": "K",  # Greek Capital Kappa
    "\u039c": "M",  # Greek Capital Mu
    "\u039d": "N",  # Greek Capital Nu
    "\u039f": "O",  # Greek Capital Omicron
    "\u03a1": "P",  # Greek Capital Rho
    "\u03a4": "T",  # Greek Capital Tau
    "\u03a5": "Y",  # Greek Capital Upsilon
    "\u03a7": "X",  # Greek Capital Chi
    "\u03b1": "a",  # Greek Small Alpha
    "\u03bf": "o",  # Greek Small Omicron
    # Fullwidth -> ASCII
    "\uff21": "A", "\uff22": "B", "\uff23": "C", "\uff24": "D",
    "\uff25": "E", "\uff26": "F", "\uff27": "G", "\uff28": "H",
    "\uff29": "I", "\uff2a": "J", "\uff2b": "K", "\uff2c": "L",
    "\uff2d": "M", "\uff2e": "N", "\uff2f": "O", "\uff30": "P",
    "\uff31": "Q", "\uff32": "R", "\uff33": "S", "\uff34": "T",
    "\uff35": "U", "\uff36": "V", "\uff37": "W", "\uff38": "X",
    "\uff39": "Y", "\uff3a": "Z",
    "\uff41": "a", "\uff42": "b", "\uff43": "c", "\uff44": "d",
    "\uff45": "e", "\uff46": "f", "\uff47": "g", "\uff48": "h",
    "\uff49": "i", "\uff4a": "j", "\uff4b": "k", "\uff4c": "l",
    "\uff4d": "m", "\uff4e": "n", "\uff4f": "o", "\uff50": "p",
    "\uff51": "q", "\uff52": "r", "\uff53": "s", "\uff54": "t",
    "\uff55": "u", "\uff56": "v", "\uff57": "w", "\uff58": "x",
    "\uff59": "y", "\uff5a": "z",
}

# Build translation table for str.translate()
_TRANS_TABLE = str.maketrans(_CONFUSABLES)


def normalize_homoglyphs(text: str) -> str:
    """
    Normalize Unicode confusables to their ASCII equivalents.

    Steps:
      1. NFKD decomposition (normalizes compatibility characters)
      2. Apply confusable lookup table
      3. Strip remaining combining marks
    """
    # Step 1: NFKD normalization
    text = unicodedata.normalize("NFKD", text)
    # Step 2: Translate known confusables
    text = text.translate(_TRANS_TABLE)
    # Step 3: Remove combining characters (accents, diacritics)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text
