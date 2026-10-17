"""
Kaggle GPU job: train + evaluate from the face crops produced by the
"Ethicura face crops" job (attached as a kernel source).

Outputs land in /kaggle/working/results: models/, results.json, figures.
"""
import glob
import os
import subprocess
import sys

SRC = "/tmp/ethicura-ai"


def sh(*cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


# Kernel outputs mount under /kaggle/input/<slug>/; find the crops wherever they are.
crops = os.path.dirname(sorted(glob.glob("/kaggle/input/**/crops/train", recursive=True))[0])
print("crops:", crops, {s: len(os.listdir(f"{crops}/{s}")) for s in os.listdir(crops)}, flush=True)

sh("git", "clone", "--depth", "1", "https://github.com/a77chand/ethicura-ai.git", SRC)
sh(sys.executable, "-m", "pip", "install", "-q", "mtcnn>=1.0", "huggingface_hub>=0.20")
for step in ["train", "evaluate"]:
    sh(sys.executable, "-m", "training.run", step, "--crops", crops,
       "--out", "/kaggle/working/results", cwd=SRC,
       env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2", "TF_XLA_FLAGS": "--tf_xla_auto_jit=0"})
