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

# Download weights in the background using Python (urllib is built-in)
python download_weights.py &

echo "Starting PDF Portal..."
python pdf_portal/app.py
