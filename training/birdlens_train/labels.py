"""Turning CUB folder names into readable common names."""

import re

# A few CUB names drop the hyphen or apostrophe from the real common name.
# I fix them by hand so the app shows proper bird names.
_SPECIAL_CASES = {
    "Chuck_will_Widow": "Chuck-will's-widow",
    "Brandt_Cormorant": "Brandt's Cormorant",
}


def clean_label(raw: str) -> str:
    """Convert "001.Black_footed_Albatross" to "Black-footed Albatross"."""
    # I drop the numeric prefix and the dot, if there is one.
    name = re.sub(r"^\d+\.", "", raw.strip())
    # I handle the names that the general rule gets wrong.
    if name in _SPECIAL_CASES:
        return _SPECIAL_CASES[name]
    words = name.split("_")
    # CUB writes hyphenated names as separate words where the second part is
    # lowercase, like "Black_footed". I join those back with a hyphen.
    parts = [words[0]]
    for word in words[1:]:
        if word and word[0].islower():
            parts[-1] = f"{parts[-1]}-{word}"
        else:
            parts.append(word)
    return " ".join(parts)
