"""
Kaggle GPU job: clone the repo and run the full training + evaluation.

Pushed with `kaggle kernels push -p training/kaggle` (see push_kernel.sh).
Outputs land in /kaggle/working/results: models/, results.json, figures,
manifest.csv.
"""
import os
import subprocess
import sys

REPO = "https://github.com/a77chand/ethicura-ai.git"
SRC = "/tmp/ethicura-ai"  # outside /kaggle/working so it isn't saved as output


def sh(*cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


sh("git", "clone", "--depth", "1", REPO, SRC)
sh(sys.executable, "-m", "pip", "install", "-q", "mtcnn>=1.0", "huggingface_hub>=0.20")
sh(sys.executable, "-c", "import tensorflow as tf, keras; print('TF', tf.__version__, 'Keras', keras.__version__, "
   "'GPUs', tf.config.list_physical_devices('GPU'))")
sh(sys.executable, "-m", "training.run", "all", "--data", "/tmp/ethicura",
   "--out", "/kaggle/working/results", "--workers", str(os.cpu_count() or 4), cwd=SRC,
   env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2", "TF_XLA_FLAGS": "--tf_xla_auto_jit=0"})
