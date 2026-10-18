"""Kaggle GPU job: PyTorch CLIP linear-probe baseline (probes/clip) on the face crops
from the "Ethicura face crops" job. Output: /kaggle/working/clip_probe.json"""
import glob
import os
import subprocess
import sys

crops = os.path.dirname(sorted(glob.glob("/kaggle/input/**/crops/train", recursive=True))[0])
subprocess.run(["git", "clone", "--depth", "1", "https://github.com/a77chand/ethicura-ai.git", "/tmp/repo"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"], check=True)
subprocess.run([sys.executable, "-m", "probes.clip.run_probe", "--crops", crops, "--cache", "/tmp/clip_cache",
                "--out", "/kaggle/working/clip_probe.json"], cwd="/tmp/repo", check=True,
               env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"})
