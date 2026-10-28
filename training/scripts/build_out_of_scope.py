"""Build the two out-of-scope photo sets used by evaluate_out_of_scope.py.

1. Non-CUB birds: research-grade iNaturalist photos of 20 species that are not
   among the 200 CUB classes, keeping only photos licensed CC0 or CC BY.
2. Non-birds: Open Images V7 validation photos (all CC BY 2.0) of animals,
   plants, objects and scenes, skipping any photo with a bird label.

The chosen photos, with their licence and author, go into manifests in
training/out_of_scope/, which I commit. The photos themselves go to the work
folder and are not committed. If a manifest already exists, I just download
the photos it lists, so the sets stay fixed. --rebuild queries the sources again.

Usage (from training/): python scripts/build_out_of_scope.py
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from birdlens_train.config import SEED, TRAINING_DIR, WORK_DIR  # noqa: E402

MANIFEST_DIR = TRAINING_DIR / "out_of_scope"
IMAGE_DIR = WORK_DIR / "out_of_scope"
PER_CLASS = 15
# The app's photo picker gives the model at most 512 px on the long side, so I
# store every photo at that size or smaller.
MAX_SIDE = 512
FIELDS = ["file", "group", "name", "source_url", "page_url", "license", "author"]

# None of these are CUB classes. I avoided the phoebes (CUB's "Sayornis" class)
# and Eurasian Tree Sparrow (CUB has American Tree Sparrow as "Tree Sparrow").
NORTH_AMERICAN = [
    "Turdus migratorius",  # American Robin
    "Zenaida macroura",  # Mourning Dove
    "Poecile atricapillus",  # Black-capped Chickadee
    "Baeolophus bicolor",  # Tufted Titmouse
    "Haemorhous mexicanus",  # House Finch
    "Ardea herodias",  # Great Blue Heron
    "Buteo jamaicensis",  # Red-tailed Hawk
    "Haliaeetus leucocephalus",  # Bald Eagle
    "Branta canadensis",  # Canada Goose
    "Sialia sialis",  # Eastern Bluebird
]
ELSEWHERE = [
    "Erithacus rubecula",  # European Robin
    "Cyanistes caeruleus",  # Eurasian Blue Tit
    "Fringilla coelebs",  # Common Chaffinch
    "Pica pica",  # Eurasian Magpie
    "Trichoglossus moluccanus",  # Rainbow Lorikeet
    "Dacelo novaeguineae",  # Laughing Kookaburra
    "Acridotheres tristis",  # Common Myna
    "Pycnonotus jocosus",  # Red-whiskered Bulbul
    "Pavo cristatus",  # Indian Peafowl
    "Malurus cyaneus",  # Superb Fairywren
]

# Open Images class names, five per group, 15 photos each.
NON_BIRD_GROUPS = {
    "animal": ["Dog", "Cat", "Horse", "Butterfly", "Fish"],
    "plant": ["Flower", "Tree", "Houseplant", "Leaf", "Cactus"],
    "object": ["Car", "Bicycle", "Chair", "Mobile phone", "Guitar"],
    # Beach and Kitchen had only 11 bird-free photos each and Street has no
    # verified labels at all, so I use Sea, Room and Road instead.
    "scene": ["Skyscraper", "Mountain", "Sea", "Room", "Road"],
}
# Any photo with a positive label containing one of these words is dropped,
# so a bird in the background can't sneak into the non-bird set.
BIRD_WORDS = [
    "bird", "duck", "goose", "swan", "owl", "eagle", "hawk", "falcon", "parrot",
    "penguin", "chicken", "rooster", "poultry", "fowl", "turkey", "ostrich",
    "pigeon", "dove", "sparrow", "gull", "heron", "flamingo", "pelican",
    "peacock", "peafowl", "crow", "raven", "finch", "robin", "woodpecker",
    "kingfisher", "feather", "beak", "egret", "stork", "crane", "cormorant",
    "wren", "warbler", "jay", "magpie", "starling", "swallow", "lorikeet",
    "macaw", "cockatoo", "toucan", "hummingbird", "vulture", "kite", "emu",
]
# Photos I dropped after looking at the set: a cartoon penguin with no bird label.
EXCLUDE_IMAGE_IDS = {"a76339c122dbfce8"}
OPEN_IMAGES = "https://storage.googleapis.com/openimages"
INAT_API = "https://api.inaturalist.org/v1/observations"


def fetch(url: str, retries: int = 5) -> bytes:
    # I retry a few times because the APIs sometimes drop a connection.
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "birdlens-eval/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def resolve_taxon(name: str) -> dict:
    # taxon_name is a fuzzy search ("Pica pica" brings back woodpeckers), so I
    # look up the exact species first and query observations by its id.
    query = urllib.parse.urlencode({"q": name, "rank": "species", "per_page": 30})
    results = json.loads(fetch(f"https://api.inaturalist.org/v1/taxa?{query}"))["results"]
    time.sleep(1.0)
    return next(t for t in results if t["name"] == name)


def inat_author(obs: dict, photo: dict) -> str:
    # CC BY photos come with a full credit line, but CC0 ones only say "no rights
    # reserved", so I add the observer's name myself (or their login if they have none).
    if photo["license_code"] != "cc0":
        return photo["attribution"]
    user = obs["user"]
    return f"{user.get('name') or user['login']}, no rights reserved (CC0)"


def build_bird_manifest(rng: np.random.Generator) -> list[dict]:
    rows = []
    for taxon in NORTH_AMERICAN + ELSEWHERE:
        species = resolve_taxon(taxon)
        query = urllib.parse.urlencode(
            {
                "taxon_id": species["id"],
                "quality_grade": "research",
                "photo_license": "cc0,cc-by",
                "photos": "true",
                "per_page": 200,
            }
        )
        results = json.loads(fetch(f"{INAT_API}?{query}"))["results"]
        # iNaturalist asks for about one request a second.
        time.sleep(1.0)
        # I keep one photo per observer so a single person can't dominate a species.
        seen_users, candidates = set(), []
        for obs in results:
            if obs["user"]["id"] in seen_users:
                continue
            photo = next((p for p in obs["photos"] if p["license_code"] in ("cc0", "cc-by")), None)
            if photo is None:
                continue
            seen_users.add(obs["user"]["id"])
            candidates.append((obs, photo))
        common = species.get("preferred_common_name", taxon)
        print(f"{common}: {len(candidates)} candidates")
        picks = rng.choice(len(candidates), size=min(PER_CLASS, len(candidates)), replace=False)
        for i in sorted(picks):
            obs, photo = candidates[i]
            rows.append(
                {
                    "file": f"inat_{photo['id']}.png",
                    "group": "north_american" if taxon in NORTH_AMERICAN else "elsewhere",
                    "name": common,
                    # The medium size is at most 500 px, close to CUB photos.
                    "source_url": photo["url"].replace("/square.", "/medium."),
                    "page_url": f"https://www.inaturalist.org/observations/{obs['id']}",
                    "license": photo["license_code"].upper().replace("CC-BY", "CC BY"),
                    "author": inat_author(obs, photo),
                }
            )
    return rows


def build_non_bird_manifest(rng: np.random.Generator, meta_dir: Path) -> list[dict]:
    # I download the Open Images metadata once into the work folder.
    files = {
        "classes": "v7/oidv7-class-descriptions.csv",
        "labels": "v7/oidv7-val-annotations-human-imagelabels.csv",
        "images": "2018_04/validation/validation-images-with-rotation.csv",
    }
    meta_dir.mkdir(parents=True, exist_ok=True)
    for rel in files.values():
        target = meta_dir / Path(rel).name
        if not target.exists():
            target.write_bytes(fetch(f"{OPEN_IMAGES}/{rel}"))
    with open(meta_dir / Path(files["classes"]).name, encoding="utf-8") as f:
        names = {row["LabelName"]: row["DisplayName"] for row in csv.DictReader(f)}
    bird_mids = {m for m, n in names.items() if any(w in n.lower() for w in BIRD_WORDS)}
    # I collect the human-verified positive labels of every validation photo.
    positives: dict[str, set[str]] = {}
    with open(meta_dir / Path(files["labels"]).name, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Confidence"] == "1":
                positives.setdefault(row["ImageID"], set()).add(row["LabelName"])
    with open(meta_dir / Path(files["images"]).name, encoding="utf-8") as f:
        images = {row["ImageID"]: row for row in csv.DictReader(f)}
    mid_of = {n: m for m, n in names.items()}
    rows, used = [], set()
    for group, class_names in NON_BIRD_GROUPS.items():
        for name in class_names:
            mid = mid_of[name]
            candidates = sorted(
                image_id
                for image_id, labels in positives.items()
                if mid in labels
                and not labels & bird_mids
                and image_id in images
                and image_id not in used
                and image_id not in EXCLUDE_IMAGE_IDS
                and "licenses/by/2.0" in images[image_id]["License"]
            )
            print(f"{name}: {len(candidates)} candidates")
            for i in sorted(rng.choice(len(candidates), size=PER_CLASS, replace=False)):
                image_id = candidates[i]
                used.add(image_id)
                meta = images[image_id]
                rows.append(
                    {
                        "file": f"oi_{image_id}.png",
                        "group": group,
                        "name": name,
                        "source_url": f"https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg",
                        "page_url": meta["OriginalLandingURL"],
                        "license": "CC BY 2.0",
                        # A few Open Images author names are still HTML escaped, like O&#x27;Hara.
                        "author": html.unescape(meta["Author"]),
                        # Open Images says how far to rotate some photos to make them upright.
                        "rotation": meta["Rotation"],
                    }
                )
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = FIELDS + (["rotation"] if "rotation" in rows[0] else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def download(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        target = out_dir / row["file"]
        if target.exists():
            continue
        image = Image.open(io.BytesIO(fetch(row["source_url"])))
        # I make the photo upright the way the app does, using EXIF first.
        image = ImageOps.exif_transpose(image).convert("RGB")
        rotation = row.get("rotation") or ""
        if rotation.strip():
            # Open Images gives an anticlockwise angle, the same as Pillow's rotate.
            image = image.rotate(float(rotation), expand=True)
        # I shrink big photos to the picker's 512 px limit and save losslessly.
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
        image.save(target, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild", nargs="*", default=None, help="query the sources again for these sets (all if empty)"
    )
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    sets = {
        "non_cub_birds": lambda: build_bird_manifest(rng),
        "non_birds": lambda: build_non_bird_manifest(rng, IMAGE_DIR / "openimages_meta"),
    }
    for name, build in sets.items():
        manifest = MANIFEST_DIR / f"{name}.csv"
        rebuild = args.rebuild is not None and (not args.rebuild or name in args.rebuild)
        if rebuild or not manifest.exists():
            write_manifest(manifest, build())
        with open(manifest, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        download(rows, IMAGE_DIR / name)
        print(f"{name}: {len(rows)} photos in {IMAGE_DIR / name}")


if __name__ == "__main__":
    main()
