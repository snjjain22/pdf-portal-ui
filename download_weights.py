"""Download model weights in the background using urllib (no external deps)."""
import os
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path("/app/prompt_based_PDF_extractor/outputs/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = [
    (
        "GroundingDINO",
        "groundingdino_swint_ogc.pth",
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
    ),
    (
        "SAM",
        "sam_vit_b_01ec64.pth",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    ),
]


def download(name: str, filename: str, url: str):
    dest = MODELS_DIR / filename
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"✓ {name} weights already present", flush=True)
        return
    print(f"⏬ Downloading {name} weights from {url}...", flush=True)
    try:
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, str(tmp))
        tmp.rename(dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"✓ {name} weights downloaded ({size_mb:.1f} MB)", flush=True)
    except Exception as e:
        print(f"✗ Failed to download {name}: {e}", flush=True)


if __name__ == "__main__":
    for name, filename, url in WEIGHTS:
        download(name, filename, url)
    print("All model weights ready.", flush=True)
