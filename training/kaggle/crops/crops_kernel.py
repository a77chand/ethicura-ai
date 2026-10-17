"""
Kaggle CPU job: download the datasets and build the face-crop TFRecords.

Its output (/kaggle/working/crops, ~2 GB) is the input of the GPU training
job, so cropping runs once and training can be re-run in minutes.
"""
import os
import subprocess
import sys

SRC = "/tmp/ethicura-ai"  # outside /kaggle/working so it isn't saved as output


def sh(*cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


sh("git", "clone", "--depth", "1", "https://github.com/a77chand/ethicura-ai.git", SRC)
sh(sys.executable, "-m", "pip", "install", "-q", "mtcnn>=1.0", "huggingface_hub>=0.20")
for step in ["download", "preprocess"]:
    sh(sys.executable, "-m", "training.run", step, "--data", "/tmp/ethicura",
       "--crops", "/kaggle/working/crops", "--out", "/kaggle/working/results",
       "--workers", str(os.cpu_count() or 4), cwd=SRC,
       env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2", "CUDA_VISIBLE_DEVICES": ""})
