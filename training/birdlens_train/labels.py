"""Turning CUB folder names into readable common names."""

import re

# CUB folder names drop the apostrophe from possessive bird names, like
# "Anna_Hummingbird" for Anna's Hummingbird. I fix those by hand so the app
# shows proper names. I checked this list against classes.txt.
_SPECIAL_CASES = {
    "Brewer_Blackbird": "Brewer's Blackbird",
    "Chuck_will_Widow": "Chuck-will's-widow",
    "Brandt_Cormorant": "Brandt's Cormorant",
    "Heermann_Gull": "Heermann's Gull",
    "Anna_Hummingbird": "Anna's Hummingbird",
    "Clark_Nutcracker": "Clark's Nutcracker",
    "Scott_Oriole": "Scott's Oriole",
    "Baird_Sparrow": "Baird's Sparrow",
    "Brewer_Sparrow": "Brewer's Sparrow",
    "Harris_Sparrow": "Harris's Sparrow",
    "Henslow_Sparrow": "Henslow's Sparrow",
    "Le_Conte_Sparrow": "LeConte's Sparrow",
    "Lincoln_Sparrow": "Lincoln's Sparrow",
    "Nelson_Sharp_tailed_Sparrow": "Nelson's Sharp-tailed Sparrow",
    "Forsters_Tern": "Forster's Tern",
    "Swainson_Warbler": "Swainson's Warbler",
    "Wilson_Warbler": "Wilson's Warbler",
    "Bewick_Wren": "Bewick's Wren",
    # Two more fixes: a hyphenated name with a capital in the middle, and a
    # spelling mistake in the original folder name.
    "Whip_poor_Will": "Whip-poor-will",
    "Artic_Tern": "Arctic Tern",
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
