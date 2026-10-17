"""
Training + evaluation pipeline for Ethicura AI's own deepfake detector, run on Modal.

    modal run training/pipeline.py::download     # fetch datasets into a Modal Volume
    modal run training/pipeline.py::inventory    # summarise what was downloaded
    modal run training/pipeline.py::preprocess   # face crops → TFRecord shards

Data lives in the `ethicura-data` Volume so every step (preprocessing,
training, evaluation) reads the same files without re-downloading.

Datasets:
  - DeepFakeFace (DFF, OpenRL/DeepFakeFace, OpenRAIL): 30k real IMDB-WIKI
    photos, each with one fake per generator — Stable Diffusion inpainting,
    InsightFace face swap, and Stable Diffusion text-to-image.
  - SDFVD (Hemgg/SDFVD-video-dataset): 53 real + 53 face-swapped videos from
    a commercial tool. Used only as an out-of-domain test set.
"""

import modal

DATA_DIR = "/data"
CROPS_DIR = f"{DATA_DIR}/crops"

# wiki = the real photos; the rest are fakes generated from them.
DFF_SOURCES = ["wiki", "inpainting", "insight", "text2img"]
FAKE_SOURCES = DFF_SOURCES[1:]
SPLITS = ["train", "val", "test"]

volume = modal.Volume.from_name("ethicura-data", create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libglib2.0-0", "unzip")
    .pip_install("huggingface_hub>=0.20.0", "numpy>=1.23.0", "Pillow>=9.5.0")
)

# Same dependencies as the API, so face cropping here is byte-for-byte the
# code path the live service runs.
ml_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libglib2.0-0")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("scikit-learn>=1.3", "matplotlib>=3.7", "pandas>=2.0")
    .env({"TF_XLA_FLAGS": "--tf_xla_auto_jit=0", "TF_CPP_MIN_LOG_LEVEL": "2"})
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App("ethicura-training")


def split_for(identity: str) -> str:
    """Deterministic person-level split (80/10/10): every photo of a person,
    and every fake made from it, lands in the same split."""
    import hashlib

    bucket = int(hashlib.md5(identity.encode()).hexdigest(), 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def is_grayscale(image_bgr) -> bool:
    if image_bgr.ndim == 2 or image_bgr.shape[2] == 1:
        return True
    b, g, r = (image_bgr[..., i].astype("int16") for i in range(3))
    return max(abs(b - g).mean(), abs(g - r).mean()) < 3.0


def make_face_cropper():
    """The API's own MTCNN face crop (20% margin, detection on a ≤320px copy),
    without loading the classifier."""
    from mtcnn import MTCNN
    from src.detector import EthicuraDetector

    detector = EthicuraDetector()
    detector._face_detector = MTCNN()
    return detector


def to_model_input(detector, image_bgr):
    """BGR image → (224x224 RGB face crop as uint8, bbox) exactly as the API
    prepares it before scaling, or (None, None) if no face is found."""
    import cv2

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    face, bbox = detector._extract_face(image_rgb)
    if face is None:
        return None, None
    return cv2.resize(face, (detector.INPUT_SIZE, detector.INPUT_SIZE)), bbox


def tf_example(face_rgb, label: int, source: str, group: str, name: str, face_px: int):
    """face_px: face box width in the *original* image — lets training drop
    tiny, upscaled-to-blur faces without re-running preprocessing."""
    import cv2
    import tensorflow as tf

    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
    feature = {
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[jpg.tobytes()])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        "source": tf.train.Feature(bytes_list=tf.train.BytesList(value=[source.encode()])),
        "group": tf.train.Feature(bytes_list=tf.train.BytesList(value=[group.encode()])),
        "name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[name.encode()])),
        "face_px": tf.train.Feature(int64_list=tf.train.Int64List(value=[face_px])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


@app.function(image=base_image, volumes={DATA_DIR: volume}, timeout=3600, cpu=4.0)
def download():
    """DeepFakeFace (DFF) zips → /data/dff/<subset>/, SDFVD videos → /data/sdfvd/."""
    import os
    import subprocess
    from huggingface_hub import hf_hub_download, snapshot_download

    for subset in ["wiki", "inpainting", "insight", "text2img"]:
        target = f"{DATA_DIR}/dff/{subset}"
        if os.path.isdir(target) and os.listdir(target):
            print(f"[dff] {subset}: already present, skipping")
            continue
        zip_path = hf_hub_download("OpenRL/DeepFakeFace", f"{subset}.zip",
                                   repo_type="dataset", local_dir="/tmp/dff")
        os.makedirs(target, exist_ok=True)
        subprocess.run(["unzip", "-q", "-o", zip_path, "-d", target], check=True)
        os.remove(zip_path)
        print(f"[dff] {subset}: extracted")
        volume.commit()

    snapshot_download("Hemgg/SDFVD-video-dataset", repo_type="dataset",
                      local_dir=f"{DATA_DIR}/sdfvd", allow_patterns=["Real/*", "Fake/*"])
    volume.commit()
    print("[sdfvd] downloaded")


@app.function(image=base_image, volumes={DATA_DIR: volume}, timeout=1800, cpu=4.0)
def inventory():
    """Print the directory layout, file counts, formats and image sizes."""
    import collections
    import os
    import random
    from PIL import Image

    for root_name in ["dff", "sdfvd"]:
        root = f"{DATA_DIR}/{root_name}"
        files = []
        for dirpath, _, filenames in os.walk(root):
            files += [os.path.join(dirpath, f) for f in filenames]
        print(f"\n== {root_name}: {len(files)} files")

        by_dir = collections.Counter(os.path.relpath(os.path.dirname(f), root) for f in files)
        for d, n in sorted(by_dir.items())[:40]:
            print(f"  {d:70s} {n}")
        if len(by_dir) > 40:
            print(f"  ... {len(by_dir)} directories total")

        exts = collections.Counter(os.path.splitext(f)[1].lower() for f in files)
        print("  extensions:", dict(exts))

        if root_name == "dff":
            top = collections.defaultdict(list)
            for f in files:
                top[os.path.relpath(f, root).split(os.sep)[0]].append(f)
            for subset, fs in sorted(top.items()):
                sample = random.Random(0).sample(fs, min(200, len(fs)))
                sizes = collections.Counter()
                fmts = collections.Counter()
                for f in sample:
                    try:
                        with Image.open(f) as im:
                            sizes[im.size] += 1
                            fmts[(im.format, im.mode)] += 1
                    except Exception as e:
                        fmts[f"error: {type(e).__name__}"] += 1
                print(f"  [{subset}] {len(fs)} files; sizes {sizes.most_common(4)}; formats {dict(fmts)}")
                print(f"     examples: {[os.path.relpath(f, root) for f in sorted(fs)[:3]]}")


# ── Preprocessing ─────────────────────────────────────────────────────────────

@app.function(image=ml_image, volumes={DATA_DIR: volume}, timeout=600)
def list_dff_names() -> list[str]:
    """Relative paths (e.g. '00/10049200_1891-09-16_1958.jpg') present in
    every DFF subset, and a clean output directory."""
    import os
    import shutil

    shutil.rmtree(CROPS_DIR, ignore_errors=True)
    for split in SPLITS + ["sdfvd"]:
        os.makedirs(f"{CROPS_DIR}/{split}", exist_ok=True)
    volume.commit()

    def names(source):
        root = f"{DATA_DIR}/dff/{source}/{source}"
        return {os.path.relpath(os.path.join(d, f), root)
                for d, _, fs in os.walk(root) for f in fs}

    common = set.intersection(*(names(s) for s in DFF_SOURCES))
    return sorted(common)


@app.function(image=ml_image, volumes={DATA_DIR: volume}, cpu=2.0, memory=4096, timeout=3600)
def preprocess_dff_chunk(names: list[str], shard: int) -> list[dict]:
    """Face-crop one chunk of DFF pairs into per-split TFRecord shards."""
    import os

    import cv2
    import tensorflow as tf

    detector = make_face_cropper()
    writers = {s: tf.io.TFRecordWriter(f"{CROPS_DIR}/{s}/dff-{shard:04d}.tfrecord") for s in SPLITS}
    rows = []
    for name in names:
        identity = os.path.basename(name).split("_")[0]
        split = split_for(identity)
        for source in DFF_SOURCES:
            row = {"name": name, "source": source, "split": split}
            image = cv2.imread(f"{DATA_DIR}/dff/{source}/{source}/{name}", cv2.IMREAD_UNCHANGED)
            if image is None:
                rows.append({**row, "status": "unreadable"})
                continue
            if is_grayscale(image):
                # Grayscale shows up unevenly across classes (old real photos,
                # some text2img fakes) — a free label cue, so drop it everywhere.
                rows.append({**row, "status": "grayscale_dropped"})
                continue
            if image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            face, bbox = to_model_input(detector, image)
            if face is None:
                rows.append({**row, "status": "no_face"})
                continue
            label = 0 if source == "wiki" else 1
            writers[split].write(tf_example(face, label, source, identity, name, bbox["w"]))
            rows.append({**row, "status": "ok", "face_w": bbox["w"], "face_h": bbox["h"],
                         "image_w": image.shape[1], "image_h": image.shape[0]})
    for w in writers.values():
        w.close()
    volume.commit()
    return rows


@app.function(image=ml_image, volumes={DATA_DIR: volume}, cpu=2.0, memory=4096, timeout=3600)
def preprocess_sdfvd_video(path: str, frames_per_video: int = 16) -> list[dict]:
    """Face crops from evenly spaced frames of one SDFVD video."""
    import os

    import cv2
    import numpy as np
    import tensorflow as tf

    detector = make_face_cropper()
    label = 1 if "/Fake/" in path else 0
    # Fake vsN.mp4 is the face-swapped version of real vN.mp4 — group them.
    video_id = os.path.splitext(os.path.basename(path))[0]
    group = video_id.lstrip("vs").lstrip("v")

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = set(np.linspace(0, max(total - 1, 0), frames_per_video).astype(int).tolist())
    writer = tf.io.TFRecordWriter(f"{CROPS_DIR}/sdfvd/{'fake' if label else 'real'}-{video_id}.tfrecord")
    kept = 0
    for idx in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        if idx not in wanted:
            continue
        face, bbox = to_model_input(detector, frame)
        if face is not None:
            writer.write(tf_example(face, label, "sdfvd", group, f"{video_id}:{idx}", bbox["w"]))
            kept += 1
    writer.close()
    cap.release()
    volume.commit()
    return [{"name": video_id, "source": "sdfvd", "split": "sdfvd", "label": label,
             "frames_total": total, "faces_kept": kept}]


@app.function(image=ml_image, volumes={DATA_DIR: volume}, timeout=600)
def write_manifest(rows: list[dict]):
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(f"{CROPS_DIR}/manifest.csv", index=False)
    volume.commit()


@app.local_entrypoint()
def preprocess(limit: int = 0):
    """--limit N: smoke-test on the first N pairs and 2 videos."""
    import collections

    names = list_dff_names.remote()
    print(f"DFF pairs present in all 4 subsets: {len(names)}")
    if limit:
        names = names[:limit]
    chunk = 500
    jobs = [(names[i:i + chunk], i // chunk) for i in range(0, len(names), chunk)]
    rows = [r for rs in preprocess_dff_chunk.starmap(jobs) for r in rs]

    videos = [f"{DATA_DIR}/sdfvd/{cls}/{f}" for cls in ["Real", "Fake"]
              for f in sdfvd_listing.remote(cls)]
    if limit:
        videos = [videos[0], videos[-1]]
    rows += [r for rs in preprocess_sdfvd_video.map(videos) for r in rs]
    write_manifest.remote(rows)

    status = collections.Counter((r["source"], r["status"]) for r in rows if "status" in r)
    for k, v in sorted(status.items()):
        print(f"  {k[0]:11s} {k[1]:24s} {v}")
    split_counts = collections.Counter((r["split"], r["source"]) for r in rows if r.get("status") == "ok")
    for k, v in sorted(split_counts.items()):
        print(f"  {k[0]:6s} {k[1]:11s} {v}")
    sd = [r for r in rows if r["source"] == "sdfvd"]
    print(f"  sdfvd videos: {len(sd)}, with ≥1 face: {sum(r['faces_kept'] > 0 for r in sd)}, "
          f"face crops: {sum(r['faces_kept'] for r in sd)}")


@app.function(image=base_image, volumes={DATA_DIR: volume})
def sdfvd_listing(cls: str) -> list[str]:
    import os

    return sorted(f for f in os.listdir(f"{DATA_DIR}/sdfvd/{cls}") if f.endswith(".mp4"))


@app.function(image=ml_image, volumes={DATA_DIR: volume}, timeout=600)
def contact_sheet(split: str = "train", n_names: int = 6) -> bytes:
    """PNG grid of face crops: one row per photo, columns real → each fake."""
    import collections
    import glob

    import cv2
    import numpy as np
    import tensorflow as tf

    spec = {"image": tf.io.FixedLenFeature([], tf.string),
            "source": tf.io.FixedLenFeature([], tf.string),
            "name": tf.io.FixedLenFeature([], tf.string)}
    by_name = collections.defaultdict(dict)
    files = sorted(glob.glob(f"{CROPS_DIR}/{split}/*.tfrecord"))
    for rec in tf.data.TFRecordDataset(files).take(4000):
        ex = tf.io.parse_single_example(rec, spec)
        by_name[ex["name"].numpy().decode()][ex["source"].numpy().decode()] = ex["image"].numpy()
    order = DFF_SOURCES if split != "sdfvd" else ["sdfvd"]
    complete = [n for n, d in by_name.items() if all(s in d for s in order)][:n_names]
    rows = []
    for n in complete:
        tiles = [cv2.imdecode(np.frombuffer(by_name[n][s], np.uint8), cv2.IMREAD_COLOR) for s in order]
        rows.append(np.hstack(tiles))
    sheet = np.vstack(rows)
    return cv2.imencode(".png", sheet)[1].tobytes()


@app.local_entrypoint()
def sheet(split: str = "train", out: str = "contact_sheet.png"):
    with open(out, "wb") as f:
        f.write(contact_sheet.remote(split))
    print(f"wrote {out}")


# ── Training ──────────────────────────────────────────────────────────────────

MODELS_DIR = f"{DATA_DIR}/models"
# Faces narrower than this in the original photo are dropped: upscaled to
# 224px they're mostly blur, and webcam faces are well above it.
MIN_FACE_PX = 80

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libglib2.0-0")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("tensorflow[and-cuda]==2.21.0", "scikit-learn>=1.3", "matplotlib>=3.7", "pandas>=2.0")
    .env({"TF_XLA_FLAGS": "--tf_xla_auto_jit=0", "TF_CPP_MIN_LOG_LEVEL": "2"})
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("training", remote_path="/root/training")
)


def predict(model, dataset):
    """Run a model over an eval_dataset → dict of numpy arrays."""
    import numpy as np

    out = {"prob": [], "label": [], "source": [], "group": [], "name": []}
    for image, label, source, group, name in dataset:
        out["prob"].append(model(image, training=False).numpy()[:, 0])
        out["label"].append(label.numpy())
        out["source"].append(source.numpy())
        out["group"].append(group.numpy())
        out["name"].append(name.numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def local_copy(split: str) -> list[str]:
    """Copy a split's shards to local disk (training streams them repeatedly)."""
    import glob
    import shutil

    shutil.copytree(f"{CROPS_DIR}/{split}", f"/tmp/crops/{split}", dirs_exist_ok=True)
    return sorted(glob.glob(f"/tmp/crops/{split}/*.tfrecord"))


@app.function(image=gpu_image, gpu="L4", cpu=8.0, memory=32768,
              volumes={DATA_DIR: volume}, timeout=4 * 3600)
def train(run_name: str = "main", exclude_source: str | None = None, epochs: int = 8,
          head_epochs: int = 1, batch_size: int = 64, lr: float = 1e-4,
          min_face_px: int = MIN_FACE_PX, seed: int = 0) -> dict:
    """Fine-tune EfficientNetB0 on DFF. exclude_source drops one generator's
    fakes from train *and* val, for leave-one-generator-out evaluation."""
    import json
    import os
    import time

    import keras
    import numpy as np
    import tensorflow as tf

    from training import data
    from training.metrics import balanced_indices, summary
    from training.model import build_model, fit_temperature, fold_temperature, logit, to_float32

    started = time.time()
    keras.utils.set_random_seed(seed)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    train_files, val_files = local_copy("train"), local_copy("val")
    fake_sources = [s for s in FAKE_SOURCES if s != exclude_source]
    n_real = int(data.records(train_files, min_face_px, ["wiki"]).reduce(0, lambda n, _: n + 1))
    steps = 2 * n_real // batch_size
    print(f"[{run_name}] fakes={fake_sources} train reals={n_real} steps/epoch={steps}")

    keras.mixed_precision.set_global_policy("mixed_float16")
    model, backbone = build_model()
    train_ds = data.training_dataset(train_files, batch_size, fake_sources, min_face_px, seed)
    val_eval = data.eval_dataset(val_files, 256, min_face_px, ["wiki"] + fake_sources)
    val_ds = val_eval.map(lambda x, y, *_: (x, y))
    auc = lambda: keras.metrics.AUC(name="auc")

    # Phase 1: new head on frozen ImageNet features.
    backbone.trainable = False
    model.compile(keras.optimizers.Adam(1e-3), "binary_crossentropy", metrics=[auc()])
    model.fit(train_ds, epochs=head_epochs, steps_per_epoch=steps, validation_data=val_ds, verbose=2)

    # Phase 2: fine-tune the whole network, keep the best epoch by val AUC.
    backbone.trainable = True
    schedule = keras.optimizers.schedules.CosineDecay(lr, decay_steps=epochs * steps, alpha=0.01)
    model.compile(keras.optimizers.AdamW(schedule, weight_decay=1e-5), "binary_crossentropy",
                  metrics=[auc()])
    best = f"/tmp/{run_name}.best.weights.h5"
    history = model.fit(
        train_ds, epochs=epochs, steps_per_epoch=steps, validation_data=val_ds, verbose=2,
        callbacks=[keras.callbacks.ModelCheckpoint(best, monitor="val_auc", mode="max",
                                                   save_best_only=True, save_weights_only=True)])
    model.load_weights(best)
    model = to_float32(model)

    # Calibrate: fit a temperature on a class-balanced slice of val, then
    # fold it into the last layer.
    val = predict(model, val_eval)
    idx = balanced_indices(val["label"], seed)
    temperature = fit_temperature(val["prob"][idx], val["label"][idx])
    fold_temperature(model, temperature)
    calibrated = 1 / (1 + np.exp(-logit(val["prob"]) / temperature))

    out_dir = f"{MODELS_DIR}/{run_name}"
    os.makedirs(out_dir, exist_ok=True)
    model.save(f"{out_dir}/model.keras")
    meta = {
        "run_name": run_name,
        "exclude_source": exclude_source,
        "fake_sources": fake_sources,
        "config": {"epochs": epochs, "head_epochs": head_epochs, "batch_size": batch_size,
                   "lr": lr, "min_face_px": min_face_px, "seed": seed,
                   "backbone": "EfficientNetB0 (ImageNet)", "input": "224x224 RGB in [0,1]"},
        "train_reals": n_real,
        "steps_per_epoch": steps,
        "temperature": temperature,
        "val_balanced": {
            "uncalibrated": summary(val["label"][idx], val["prob"][idx]),
            "calibrated": summary(val["label"][idx], calibrated[idx]),
        },
        "history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
        "best_epoch": int(np.argmax(history.history["val_auc"])) + 1,
        "train_minutes": round((time.time() - started) / 60, 1),
    }
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    volume.commit()
    print(f"[{run_name}] done: best epoch {meta['best_epoch']}, T={temperature:.3f}, "
          f"val AUC {meta['val_balanced']['calibrated']['auc']:.4f}, {meta['train_minutes']} min")
    return meta


@app.local_entrypoint()
def train_all(epochs: int = 8, only_main: bool = False):
    """Main model + three leave-one-generator-out models, in parallel."""
    runs = [("main", None)]
    if not only_main:
        runs += [(f"logo-{s}", s) for s in FAKE_SOURCES]
    for meta in train.starmap(runs, kwargs={"epochs": epochs}):
        cal = meta["val_balanced"]["calibrated"]
        print(f"{meta['run_name']:18s} best epoch {meta['best_epoch']}  T={meta['temperature']:.3f}  "
              f"val AUC {cal['auc']:.4f}  EER {cal['eer']:.4f}  ECE {cal['ece']:.4f}  "
              f"({meta['train_minutes']} min)")


@app.function(image=ml_image, volumes={DATA_DIR: volume}, timeout=600)
def manifest_stats() -> str:
    import pandas as pd

    df = pd.read_csv(f"{CROPS_DIR}/manifest.csv")
    lines = [df.groupby(["source", "status"]).size().to_string()]
    ok = df[df.status == "ok"]
    lines.append("\nface width (px in original image) quantiles:")
    lines.append(ok.groupby("source").face_w.quantile([0.05, 0.25, 0.5, 0.75]).unstack().round(0).to_string())
    for t in [64, 80, 100, 120]:
        kept = ok[ok.face_w >= t].groupby(["split", "source"]).size().unstack()
        lines.append(f"\nkept with face_w >= {t}:\n{kept.to_string()}")
    return "\n".join(lines)


@app.local_entrypoint()
def stats():
    print(manifest_stats.remote())


# ── Evaluation ────────────────────────────────────────────────────────────────

BASELINE_REPO = "ariyul/deepfake-detection-cnn_v2"


def load_baseline():
    """The third-party pretrained model the live demo used before this work."""
    import keras
    from huggingface_hub import snapshot_download

    path = snapshot_download(BASELINE_REPO)
    return keras.saving.load_model(f"{path}/keras_model/best_model.keras")


def vs_source(pred, source: str, real_source: str = "wiki"):
    """Mask selecting real photos plus one generator's fakes."""
    import numpy as np

    src = np.char.decode(pred["source"])
    return (src == real_source) | (src == source)


@app.function(image=gpu_image, gpu="L4", cpu=8.0, memory=32768,
              volumes={DATA_DIR: volume}, timeout=3 * 3600)
def evaluate(min_face_px: int = MIN_FACE_PX) -> dict:
    """Held-out evaluation of the trained model against the third-party
    baseline: overall, per generator, under degradation, on an unseen
    generator, and on real face-swap *videos* from a different dataset."""
    import json
    import os

    import keras
    import numpy as np

    from training import data
    from training.metrics import balanced_indices, bootstrap_auc_ci, reliability, summary

    test_files = local_copy("test")
    sdfvd_files = sorted(f"{CROPS_DIR}/sdfvd/{f}" for f in os.listdir(f"{CROPS_DIR}/sdfvd")
                         if f.endswith(".tfrecord"))
    models = {"ethicura": keras.saving.load_model(f"{MODELS_DIR}/main/model.keras", compile=False),
              "baseline": load_baseline()}
    results = {"min_face_px": min_face_px, "models": {}}
    curves = {}

    def run(model, files, transform=None, sources=None, min_px=min_face_px):
        return predict(model, data.eval_dataset(files, 256, min_px, sources, transform))

    for name, model in models.items():
        r = {}
        clean = run(model, test_files)
        labels, probs, groups = clean["label"].astype(int), clean["prob"], np.char.decode(clean["group"])
        r["overall"] = summary(labels, probs)
        r["overall"]["auc_ci95"] = bootstrap_auc_ci(labels, probs, groups)
        bal = balanced_indices(labels)
        r["overall_balanced"] = summary(labels[bal], probs[bal])
        r["reliability"] = reliability(labels[bal], probs[bal])
        curves[name] = (labels, probs)

        # Per generator: real photos vs that generator's fakes only.
        r["per_generator"] = {}
        for s in FAKE_SOURCES:
            m = vs_source(clean, s)
            sm = summary(labels[m], probs[m])
            sm["auc_ci95"] = bootstrap_auc_ci(labels[m], probs[m], groups[m], n_boot=500)
            r["per_generator"][s] = sm

        # Robustness: same test set under compression / downscale / blur.
        r["robustness"] = {}
        for pname, fn in data.PERTURBATIONS.items():
            p = run(model, test_files, fn)
            r["robustness"][pname] = summary(p["label"].astype(int), p["prob"])
        results["models"][name] = r
        print(f"[{name}] test AUC {r['overall']['auc']:.4f}  "
              f"bal-acc {r['overall']['balanced_accuracy']:.4f}  ECE {r['overall_balanced']['ece']:.4f}")

        # Out-of-domain: face-swapped *videos* from a different source.
        sd = run(model, sdfvd_files, min_px=0)
        sl, sp, sg = sd["label"].astype(int), sd["prob"], np.char.decode(sd["group"])
        frame = summary(sl, sp)
        frame["auc_ci95"] = bootstrap_auc_ci(sl, sp, sg, n_boot=500)
        names = np.char.decode(sd["name"])
        vid = {}
        for n, p, l in zip(names, sp, sl):
            vid.setdefault(n.split(":")[0], []).append((p, l))
        vids = [(np.mean([p for p, _ in v]), v[0][1], k) for k, v in vid.items()]
        vl = np.array([l for _, l, _ in vids]); vp = np.array([p for p, _, _ in vids])
        vg = np.array([k.lstrip("vs").lstrip("v") for _, _, k in vids])
        video = summary(vl, vp)
        video["auc_ci95"] = bootstrap_auc_ci(vl, vp, vg, n_boot=500)
        results["models"][name]["sdfvd_frame"] = frame
        results["models"][name]["sdfvd_video"] = video
        print(f"[{name}] SDFVD frame AUC {frame['auc']:.4f}  video AUC {video['auc']:.4f} (n={len(vids)})")

    # Leave-one-generator-out: models that never saw generator s, tested on it.
    results["leave_one_generator_out"] = {}
    main = models["ethicura"]
    for s in FAKE_SOURCES:
        path = f"{MODELS_DIR}/logo-{s}/model.keras"
        if not os.path.exists(path):
            continue
        held_out = keras.saving.load_model(path, compile=False)
        entry = {}
        for tag, model in [("held_out_model", held_out), ("full_model", main)]:
            p = run(model, test_files, sources=["wiki", s])
            labels, probs, groups = p["label"].astype(int), p["prob"], np.char.decode(p["group"])
            entry[tag] = summary(labels, probs)
            entry[tag]["auc_ci95"] = bootstrap_auc_ci(labels, probs, groups, n_boot=500)
        # Same held-out model on the generators it *did* see, for contrast.
        seen = [x for x in FAKE_SOURCES if x != s]
        p = run(held_out, test_files, sources=["wiki"] + seen)
        entry["held_out_model_on_seen_generators"] = summary(p["label"].astype(int), p["prob"])
        results["leave_one_generator_out"][s] = entry
        print(f"[logo-{s}] unseen AUC {entry['held_out_model']['auc']:.4f} "
              f"(seen-trained {entry['full_model']['auc']:.4f})")

    os.makedirs("/tmp/out", exist_ok=True)
    figures = render_figures(results, curves)
    return {"results": results, "figures": figures}


def render_figures(results, curves) -> dict:
    """ROC curves, reliability diagram and robustness chart → {filename: png bytes}."""
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    colors = {"ethicura": "#d9534f", "baseline": "#6b7280"}
    labels_ = {"ethicura": "Ethicura (this work)", "baseline": "Third-party baseline"}
    figs = {}

    def png(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        auc = results["models"][name]["overall"]["auc"]
        ax.plot(fpr, tpr, color=colors[name], label=f"{labels_[name]}  AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], ":", color="#bbb")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (fakes caught)",
           title="ROC, held-out people", xscale="log", xlim=(1e-3, 1))
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    figs["roc.png"] = png(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ":", color="#bbb", label="perfectly calibrated")
    for name, r in results["models"].items():
        pts = r["reliability"]
        ax.plot([a for a, _, _ in pts], [b for _, b, _ in pts], "o-", color=colors[name],
                label=f"{labels_[name]}  ECE {r['overall_balanced']['ece']:.3f}")
    ax.set(xlabel="Predicted P(fake)", ylabel="Observed fraction fake",
           title="Calibration (50/50 balanced test set)")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    figs["calibration.png"] = png(fig)

    names = list(next(iter(results["models"].values()))["robustness"])
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.38
    for i, (name, r) in enumerate(results["models"].items()):
        ax.bar([j + (i - 0.5) * width for j in range(len(names))],
               [r["robustness"][n]["auc"] for n in names], width, color=colors[name], label=labels_[name])
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set(ylabel="AUC", ylim=(0.4, 1.0), title="Robustness: AUC under degradation")
    ax.axhline(0.5, color="#bbb", ls=":"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    figs["robustness.png"] = png(fig)
    return figs


@app.local_entrypoint()
def evaluate_all(out: str = "results"):
    import json
    import os

    payload = evaluate.remote()
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/results.json", "w") as f:
        json.dump(payload["results"], f, indent=2)
    for fname, content in payload["figures"].items():
        with open(f"{out}/{fname}", "wb") as f:
            f.write(content)
    print(f"wrote {out}/results.json and {len(payload['figures'])} figures")
