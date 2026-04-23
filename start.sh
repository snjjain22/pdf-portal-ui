#!/bin/bash
set -e

# Models are stored inside the volume-backed outputs/ dir so they survive redeploys
MODELS_DIR="/app/prompt_based_PDF_extractor/outputs/models"
MODELS_LINK="/app/prompt_based_PDF_extractor/models"

mkdir -p "$MODELS_DIR"

# Symlink models/ → outputs/models/ so pipeline code finds weights at expected path
if [ ! -L "$MODELS_LINK" ]; then
    rm -rf "$MODELS_LINK"
    ln -s "$MODELS_DIR" "$MODELS_LINK"
    echo "✓ Symlinked models/ → outputs/models/"
fi

# Download GroundingDINO weights if not already present
DINO_WEIGHTS="$MODELS_DIR/groundingdino_swint_ogc.pth"
if [ ! -f "$DINO_WEIGHTS" ]; then
    echo "Downloading GroundingDINO weights (~700MB)..."
    wget -q --show-progress -O "$DINO_WEIGHTS" \
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    echo "✓ GroundingDINO weights downloaded"
else
    echo "✓ GroundingDINO weights already present"
fi

# Download SAM weights if not already present
SAM_WEIGHTS="$MODELS_DIR/sam_vit_b_01ec64.pth"
if [ ! -f "$SAM_WEIGHTS" ]; then
    echo "Downloading SAM weights (~375MB)..."
    wget -q --show-progress -O "$SAM_WEIGHTS" \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    echo "✓ SAM weights downloaded"
else
    echo "✓ SAM weights already present"
fi

echo "Starting PDF Portal..."
python pdf_portal/app.py
