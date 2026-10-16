"""
Deploy the Ethicura AI API on Modal (https://modal.com).

    pip install modal
    modal token new          # one-time browser login
    modal deploy modal_app.py

Serves the same FastAPI app as `uvicorn src.api:app`. Containers scale to
zero when idle; the first request after that waits for a cold start, which
the frontend shows as "waking server" by polling /health.
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libglib2.0-0")
    .pip_install_from_requirements("requirements.txt")
    .env({"TF_XLA_FLAGS": "--tf_xla_auto_jit=0"})
    # Bake the pretrained classifier into the image so a cold start doesn't
    # re-download it from the Hub.
    .run_commands(
        "python -c \"from huggingface_hub import snapshot_download; "
        "snapshot_download('ariyul/deepfake-detection-cnn_v2')\""
    )
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App("ethicura-ai-api", image=image)


@app.function(
    cpu=2.0,
    memory=4096,
    scaledown_window=300,  # stay warm 5 min after the last request
    max_containers=2,      # caps spend; one container handles a demo fine
    timeout=600,
)
# Lets /health answer while a frame is being processed; inference itself is
# serialised by the detector's lock.
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def api():
    from src.api import app as fastapi_app

    return fastapi_app
