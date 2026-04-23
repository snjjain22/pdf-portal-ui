FROM python:3.10-slim

# Build tools needed for GroundingDINO + SAM compilation
RUN apt-get update && apt-get install -y \
    gcc g++ git ninja-build \
    poppler-utils \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy entire repo
COPY . .

# Install CPU-only PyTorch first (avoids pulling CUDA binaries)
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install GroundingDINO and SAM
RUN pip install --no-cache-dir groundingdino-py>=0.1.0 segment-anything>=1.0

# Install all pipeline + portal dependencies
RUN pip install --no-cache-dir -r prompt_based_PDF_extractor/requirements.txt
RUN pip install --no-cache-dir -r pdf_portal/requirements.txt

# Tell the portal where the repo root is inside the container
ENV REPO_ROOT=/app

# Expose Gradio port
EXPOSE 7860

CMD ["python", "pdf_portal/app.py"]
