FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Bake the pretrained classifier into the image so a cold start doesn't
# re-download it from the Hub.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('ariyul/deepfake-detection-cnn_v2')"

COPY --chown=user src ./src

# 7860 is the port Hugging Face Spaces expects; other hosts can override PORT.
ENV PORT=7860
EXPOSE 7860
CMD uvicorn src.api:app --host 0.0.0.0 --port ${PORT}
