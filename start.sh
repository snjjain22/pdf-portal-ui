#!/bin/bash

MODELS_DIR="/app/prompt_based_PDF_extractor/outputs/models"
MODELS_LINK="/app/prompt_based_PDF_extractor/models"

mkdir -p "$MODELS_DIR"

# Symlink models/ → outputs/models/ so pipeline finds weights at expected path
if [ ! -L "$MODELS_LINK" ]; then
    rm -rf "$MODELS_LINK"
    ln -s "$MODELS_DIR" "$MODELS_LINK"
    echo "✓ Symlinked models/ → outputs/models/"
fi

# Download weights in the background so the portal starts immediately
download_weights() {
    DINO_WEIGHTS="$MODELS_DIR/groundingdino_swint_ogc.pth"
    if [ ! -f "$DINO_WEIGHTS" ]; then
        echo "⏬ Downloading GroundingDINO weights (~700MB) in background..."
        curl -L --silent -o "$DINO_WEIGHTS" \
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" \
            && echo "✓ GroundingDINO weights ready" \
            || echo "✗ Failed to download GroundingDINO weights"
    else
        echo "✓ GroundingDINO weights already present"
    fi

    SAM_WEIGHTS="$MODELS_DIR/sam_vit_b_01ec64.pth"
    if [ ! -f "$SAM_WEIGHTS" ]; then
        echo "⏬ Downloading SAM weights (~375MB) in background..."
        curl -L --silent -o "$SAM_WEIGHTS" \
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
            && echo "✓ SAM weights ready" \
            || echo "✗ Failed to download SAM weights"
    else
        echo "✓ SAM weights already present"
    fi
}

download_weights &

echo "Starting PDF Portal..."
python pdf_portal/app.py
