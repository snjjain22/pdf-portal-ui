FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    gcc g++ git ninja-build \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Pin numpy below 2 and scipy below 1.14 (both needed for numpy-1.x compatibility)
RUN pip install --no-cache-dir 'numpy<2' 'scipy<1.14'

# Install torch CPU version (pinned exactly)
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install transformers/tokenizers/HF hub at versions compatible with torch 2.2
# Also pin starlette<1.0 and fastapi<0.115 (gradio 4.44.1 needs old TemplateResponse API)
RUN pip install --no-cache-dir \
    transformers==4.44.2 tokenizers==0.19.1 huggingface-hub==0.24.7 \
    'starlette<1.0' 'fastapi<0.115'

# Install GroundingDINO + SAM with --no-deps so they don't downgrade torch/transformers
RUN pip install --no-cache-dir --no-deps groundingdino-py segment-anything

# Install GroundingDINO's actual runtime deps manually
RUN pip install --no-cache-dir --no-deps addict yapf timm supervision pycocotools
RUN pip install --no-cache-dir omegaconf opencv-python-headless

# Install pipeline and portal requirements using the constraints file
RUN pip install --no-cache-dir -c constraints.txt -r prompt_based_PDF_extractor/requirements.txt
RUN pip install --no-cache-dir -c constraints.txt -r pdf_portal/requirements.txt

# Verify versions at build time (fails build if mismatched)
RUN python -c "import torch, transformers; \
    assert torch.__version__.startswith('2.2'), f'torch={torch.__version__}'; \
    assert transformers.__version__ == '4.44.2', f'transformers={transformers.__version__}'; \
    print(f'✓ torch={torch.__version__}, transformers={transformers.__version__}')"

ENV REPO_ROOT=/app
EXPOSE 7860

CMD ["bash", "start.sh"]
