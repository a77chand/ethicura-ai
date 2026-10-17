"""
Train and evaluate Ethicura's own deepfake detector. Runs anywhere with
Python + TensorFlow (a Kaggle/Colab GPU notebook, a workstation):

    python -m training.run all        --data /tmp/ethicura --out results
or step by step:
    python -m training.run download   --data /tmp/ethicura
    python -m training.run preprocess --data /tmp/ethicura --out results [--limit N]
    python -m training.run train      --data /tmp/ethicura --out results [--runs main logo-insight ...]
    python -m training.run evaluate   --data /tmp/ethicura --out results

Each step skips work whose output already exists, so a re-run resumes.

Datasets:
  - DeepFakeFace (DFF, OpenRL/DeepFakeFace, OpenRAIL licence): 30k real
    IMDB-WIKI photos, each with one fake per generator — Stable Diffusion
    inpainting, InsightFace face swap, Stable Diffusion text-to-image.
  - SDFVD (Hemgg/SDFVD-video-dataset): 53 real + 53 face-swapped videos from
    a commercial tool. Out-of-domain test set only.

Shortcut controls (see README "Evaluation"):
  - grayscale images are dropped from every class (they occur unevenly);
  - every crop goes through the API's own MTCNN face crop and is re-encoded
    identically, removing the real photos' varied sizes/formats;
  - faces under MIN_FACE_PX in the original image are dropped, and training
    applies blur/downscale/JPEG augmentation to both classes;
  - splits are by person, so no identity appears in two splits.
"""

import argparse
import hashlib
import json
import os
import sys
import time

DFF_SOURCES = ["wiki", "inpainting", "insight", "text2img"]  # wiki = real photos
FAKE_SOURCES = DFF_SOURCES[1:]
SPLITS = ["train", "val", "test"]
MIN_FACE_PX = 80
EVAL_BATCH = 64
BASELINE_REPO = "ariyul/deepfake-detection-cnn_v2"
RUNS = {"main": None, **{f"logo-{s}": s for s in FAKE_SOURCES}}


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


# ── Download ──────────────────────────────────────────────────────────────────

def download(data: str):
    import zipfile

    from huggingface_hub import hf_hub_download, snapshot_download

    for source in DFF_SOURCES:
        target = f"{data}/dff/{source}"
        if os.path.isdir(target) and os.listdir(target):
            log(f"[download] {source}: present")
            continue
        zip_path = hf_hub_download("OpenRL/DeepFakeFace", f"{source}.zip", repo_type="dataset",
                                   local_dir=f"{data}/zips")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(target)
        os.remove(zip_path)
        log(f"[download] {source}: extracted")
    snapshot_download("Hemgg/SDFVD-video-dataset", repo_type="dataset",
                      local_dir=f"{data}/sdfvd", allow_patterns=["Real/*", "Fake/*"])
    log("[download] sdfvd: present")


# ── Preprocessing ─────────────────────────────────────────────────────────────

def split_for(identity: str) -> str:
    """Deterministic person-level split (80/10/10)."""
    bucket = int(hashlib.md5(identity.encode()).hexdigest(), 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def is_grayscale(image_bgr) -> bool:
    if image_bgr.ndim == 2 or image_bgr.shape[2] == 1:
        return True
    b, g, r = (image_bgr[..., i].astype("int16") for i in range(3))
    return max(abs(b - g).mean(), abs(g - r).mean()) < 3.0


def _worker_init():
    # Face cropping is CPU work; keep worker processes off the GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["OMP_NUM_THREADS"] = "1"
    # One TF thread per worker: the pool already uses every core, and N
    # workers each spawning N intra-op threads just fight over them.
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    import cv2
    cv2.setNumThreads(1)
    global _CROPPER
    from mtcnn import MTCNN
    from src.detector import EthicuraDetector

    # The API's own crop code (20% margin, detection on a ≤320px copy),
    # without loading the classifier.
    _CROPPER = EthicuraDetector()
    _CROPPER._face_detector = MTCNN()


def _to_model_input(image_bgr):
    """BGR image → (224x224 RGB uint8 face crop, bbox), exactly as the API
    builds it before scaling to [0, 1]; (None, None) if no face."""
    import cv2

    face, bbox = _CROPPER._extract_face(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if face is None:
        return None, None
    return cv2.resize(face, (_CROPPER.INPUT_SIZE, _CROPPER.INPUT_SIZE)), bbox


def _example(face_rgb, label, source, group, name, face_px):
    import cv2
    import tensorflow as tf

    jpg = cv2.imencode(".jpg", cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])[1]
    b = lambda v: tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
    i = lambda v: tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))
    return tf.train.Example(features=tf.train.Features(feature={
        "image": b(jpg.tobytes()), "label": i(label), "source": b(source.encode()),
        "group": b(group.encode()), "name": b(name.encode()), "face_px": i(int(face_px)),
    })).SerializeToString()


def _crop_dff_chunk(args):
    data, crops, names, shard = args
    import cv2
    import tensorflow as tf

    writers = {s: tf.io.TFRecordWriter(f"{crops}/{s}/dff-{shard:04d}.tfrecord.tmp") for s in SPLITS}
    rows = []
    for name in names:
        identity = os.path.basename(name).split("_")[0]
        split = split_for(identity)
        for source in DFF_SOURCES:
            row = {"name": name, "source": source, "split": split}
            image = cv2.imread(f"{data}/dff/{source}/{source}/{name}", cv2.IMREAD_UNCHANGED)
            if image is None:
                rows.append({**row, "status": "unreadable"})
                continue
            if is_grayscale(image):
                rows.append({**row, "status": "grayscale_dropped"})
                continue
            if image.ndim == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            try:
                face, bbox = _to_model_input(image)
            except Exception as e:  # one bad image shouldn't sink a 2-hour run
                rows.append({**row, "status": f"error: {type(e).__name__}"})
                continue
            if face is None:
                rows.append({**row, "status": "no_face"})
                continue
            writers[split].write(_example(face, int(source != "wiki"), source, identity, name, bbox["w"]))
            rows.append({**row, "status": "ok", "face_w": bbox["w"], "face_h": bbox["h"]})
    for s, w in writers.items():
        w.close()
        os.replace(f"{crops}/{s}/dff-{shard:04d}.tfrecord.tmp", f"{crops}/{s}/dff-{shard:04d}.tfrecord")
    return rows


def _crop_sdfvd_video(args):
    crops, path, frames_per_video = args
    import cv2
    import numpy as np
    import tensorflow as tf

    label = int("/Fake/" in path)
    video_id = os.path.splitext(os.path.basename(path))[0]
    group = video_id.lstrip("vs")  # fake vsN is the face-swapped version of real vN
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = set(np.linspace(0, max(total - 1, 0), frames_per_video).astype(int).tolist())
    out = f"{crops}/sdfvd/{'fake' if label else 'real'}-{video_id}.tfrecord"
    kept = 0
    with tf.io.TFRecordWriter(out) as writer:
        for idx in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            if idx in wanted:
                face, bbox = _to_model_input(frame)
                if face is not None:
                    writer.write(_example(face, label, "sdfvd", group, f"{video_id}:{idx}", bbox["w"]))
                    kept += 1
    cap.release()
    return [{"name": video_id, "source": "sdfvd", "split": "sdfvd", "label": label,
             "frames_total": total, "faces_kept": kept}]


def preprocess(data: str, crops: str, out: str, workers: int, limit: int = 0, chunk: int = 250):
    import collections
    import multiprocessing as mp

    import pandas as pd

    manifest_path = f"{out}/manifest.csv"
    if os.path.exists(manifest_path) and not limit:
        log("[preprocess] manifest exists, skipping")
        return
    for s in SPLITS + ["sdfvd"]:
        os.makedirs(f"{crops}/{s}", exist_ok=True)

    def names(source):
        root = f"{data}/dff/{source}/{source}"
        return {os.path.relpath(os.path.join(d, f), root) for d, _, fs in os.walk(root) for f in fs}

    common = sorted(set.intersection(*(names(s) for s in DFF_SOURCES)))
    if limit:
        common = common[:limit]
    jobs = [(data, crops, common[i:i + chunk], i // chunk) for i in range(0, len(common), chunk)]
    videos = sorted(f"{data}/sdfvd/{c}/{f}" for c in ["Real", "Fake"]
                    for f in os.listdir(f"{data}/sdfvd/{c}") if f.endswith(".mp4"))
    if limit:
        videos = [videos[0], videos[-1]]
    log(f"[preprocess] {len(common)} DFF pairs in {len(jobs)} chunks, {len(videos)} videos, {workers} workers")

    rows = []
    with mp.get_context("spawn").Pool(workers, initializer=_worker_init) as pool:
        for n, r in enumerate(pool.imap_unordered(_crop_dff_chunk, jobs), 1):
            rows += r
            if n % 10 == 0 or n == len(jobs):
                log(f"[preprocess] DFF chunks {n}/{len(jobs)}")
        for r in pool.imap_unordered(_crop_sdfvd_video, [(crops, v, 16) for v in videos]):
            rows += r
    os.makedirs(out, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(manifest_path, index=False)

    dff = df[df.source != "sdfvd"]
    log("[preprocess] status by source:\n" + dff.groupby(["source", "status"]).size().to_string())
    ok = dff[dff.status == "ok"]
    log("[preprocess] face width quantiles (px):\n" +
        ok.groupby("source").face_w.quantile([0.1, 0.5, 0.9]).unstack().round(0).to_string())
    kept = ok[ok.face_w >= MIN_FACE_PX].groupby(["split", "source"]).size().unstack()
    log(f"[preprocess] kept with face_w >= {MIN_FACE_PX}:\n{kept.to_string()}")
    sd = df[df.source == "sdfvd"]
    log(f"[preprocess] SDFVD: {len(sd)} videos, {int(sd.faces_kept.sum())} face crops")


# ── Training ──────────────────────────────────────────────────────────────────

def predict(model, dataset):
    import numpy as np

    out = {"prob": [], "label": [], "source": [], "group": [], "name": []}
    for image, label, source, group, name in dataset:
        out["prob"].append(model(image, training=False).numpy()[:, 0])
        out["label"].append(label.numpy())
        out["source"].append(source.numpy())
        out["group"].append(group.numpy())
        out["name"].append(name.numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def shards(crops: str, split: str) -> list[str]:
    import glob

    return sorted(glob.glob(f"{crops}/{split}/*.tfrecord"))


def train(crops: str, out: str, run_name: str, epochs: int, head_epochs: int = 1,
          batch_size: int = 64, lr: float = 1e-4, min_face_px: int = MIN_FACE_PX, seed: int = 0):
    """Fine-tune EfficientNetB0. logo-<generator> runs leave that generator's
    fakes out of train *and* val, to test generalisation to it."""
    import keras
    import numpy as np
    import tensorflow as tf

    from training import data as tdata
    from training.metrics import balanced_indices, summary
    from training.model import build_model, fit_temperature, fold_temperature, logit, to_float32

    out_dir = f"{out}/models/{run_name}"
    if os.path.exists(f"{out_dir}/meta.json"):
        log(f"[train:{run_name}] already trained, skipping")
        return
    started = time.time()
    keras.utils.set_random_seed(seed)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # already initialised by an earlier run in this process
    log(f"[train:{run_name}] GPUs: {gpus}")

    exclude = RUNS[run_name]
    fake_sources = [s for s in FAKE_SOURCES if s != exclude]
    train_files, val_files = shards(crops, "train"), shards(crops, "val")
    n_real = int(tdata.records(train_files, min_face_px, ["wiki"]).reduce(0, lambda n, _: n + 1))
    steps = 2 * n_real // batch_size
    log(f"[train:{run_name}] fakes={fake_sources} train reals={n_real} steps/epoch={steps}")

    if gpus:
        keras.mixed_precision.set_global_policy("mixed_float16")
    model, backbone = build_model()
    train_ds = tdata.training_dataset(train_files, batch_size, fake_sources, min_face_px, seed)
    val_eval = tdata.eval_dataset(val_files, EVAL_BATCH, min_face_px, ["wiki"] + fake_sources)
    val_ds = val_eval.map(lambda x, y, *_: (x, y))
    auc = lambda: keras.metrics.AUC(name="auc")

    # Phase 1: new head on frozen ImageNet features.
    backbone.trainable = False
    model.compile(keras.optimizers.Adam(1e-3), "binary_crossentropy", metrics=[auc()])
    model.fit(train_ds, epochs=head_epochs, steps_per_epoch=steps, validation_data=val_ds, verbose=2)

    # Phase 2: fine-tune everything; keep the best epoch by validation AUC.
    backbone.trainable = True
    schedule = keras.optimizers.schedules.CosineDecay(lr, decay_steps=epochs * steps, alpha=0.01)
    model.compile(keras.optimizers.AdamW(schedule, weight_decay=1e-5), "binary_crossentropy", metrics=[auc()])
    best = f"/tmp/{run_name}.best.weights.h5"
    history = model.fit(
        train_ds, epochs=epochs, steps_per_epoch=steps, validation_data=val_ds, verbose=2,
        callbacks=[keras.callbacks.ModelCheckpoint(best, monitor="val_auc", mode="max",
                                                   save_best_only=True, save_weights_only=True)])
    model.load_weights(best)
    trained = model
    keras.mixed_precision.set_global_policy("float32")
    model = to_float32(trained)
    # Free the mixed-precision training copy (and its optimizer state)
    # before running inference — holding both ran a T4 out of memory.
    del trained, train_ds
    import gc
    gc.collect()

    os.makedirs(out_dir, exist_ok=True)
    model.save(f"{out_dir}/model.keras")
    meta = {
        "run_name": run_name, "exclude_source": exclude, "fake_sources": fake_sources,
        "config": {"epochs": epochs, "head_epochs": head_epochs, "batch_size": batch_size, "lr": lr,
                   "min_face_px": min_face_px, "seed": seed, "backbone": "EfficientNetB0 (ImageNet)",
                   "input": "224x224 RGB in [0,1]", "mixed_precision": bool(gpus)},
        "train_reals": n_real, "steps_per_epoch": steps, "temperature": 1.0,
        "history": {k: [float(v) for v in vs] for k, vs in history.history.items()},
        "best_epoch": int(np.argmax(history.history["val_auc"])) + 1,
        "train_minutes": round((time.time() - started) / 60, 1),
    }
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    keras.mixed_precision.set_global_policy("float32")
    log(f"[train:{run_name}] best epoch {meta['best_epoch']}, "
        f"val AUC {max(meta['history']['val_auc']):.4f} ({meta['train_minutes']} min)")
    calibrate(crops, out, run_name, min_face_px, seed)


def calibrate(crops: str, out: str, run_name: str, min_face_px: int = MIN_FACE_PX, seed: int = 0):
    """Fit a temperature on a class-balanced slice of validation and fold it
    into the last layer. Reloads the model from disk so calibration is fitted
    to exactly the artifact that ships (an in-memory pass inside training once
    disagreed with the saved file). Re-running refines rather than compounds:
    the stored temperature is the cumulative one."""
    import keras
    import numpy as np

    from training import data as tdata
    from training.metrics import balanced_indices, summary
    from training.model import fit_temperature, fold_temperature, logit

    out_dir = f"{out}/models/{run_name}"
    with open(f"{out_dir}/meta.json") as f:
        meta = json.load(f)
    model = keras.saving.load_model(f"{out_dir}/model.keras", compile=False)
    val = predict(model, tdata.eval_dataset(shards(crops, "val"), EVAL_BATCH, min_face_px,
                                            ["wiki"] + meta["fake_sources"]))
    idx = balanced_indices(val["label"], seed)
    y, p = val["label"][idx].astype(int), val["prob"][idx]
    t = fit_temperature(p, y)
    fold_temperature(model, t)
    model.save(f"{out_dir}/model.keras")

    meta["temperature"] = meta.get("temperature", 1.0) * t
    meta["val_balanced"] = {"before": summary(y, p), "calibrated": summary(y, 1 / (1 + np.exp(-logit(p) / t)))}
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    cal = meta["val_balanced"]
    log(f"[calibrate:{run_name}] T={meta['temperature']:.3f}, val AUC {cal['calibrated']['auc']:.4f}, "
        f"ECE {cal['before']['ece']:.4f} -> {cal['calibrated']['ece']:.4f}")


# ── Evaluation ────────────────────────────────────────────────────────────────

def load_baseline():
    """The third-party model, fed the input it actually works with.

    Its model card (and this repo's original API) divide pixels by 255, but
    Keras' EfficientNet already rescales internally, so [0, 1] input gets
    squashed twice and the model outputs ~0.497 for every face. Wrapping it
    with ×255 gives it 0-255 pixels, which is the fair comparison."""
    import keras
    from huggingface_hub import snapshot_download

    path = snapshot_download(BASELINE_REPO)
    inner = keras.saving.load_model(f"{path}/keras_model/best_model.keras", compile=False)
    inputs = keras.Input((224, 224, 3))
    return keras.Model(inputs, inner(keras.layers.Rescaling(255.0)(inputs)), name="baseline_0_255")


def evaluate(crops: str, out: str, min_face_px: int = MIN_FACE_PX):
    """Held-out comparison of the trained model and the third-party baseline:
    overall, per generator, under degradation, on unseen generators, and on
    face-swap videos from a different dataset."""
    import keras
    import numpy as np

    from training import data as tdata
    from training.metrics import balanced_indices, bootstrap_auc_ci, reliability, summary

    test_files, sdfvd_files = shards(crops, "test"), shards(crops, "sdfvd")
    models = {"ethicura": keras.saving.load_model(f"{out}/models/main/model.keras", compile=False),
              "baseline": load_baseline()}
    results = {"min_face_px": min_face_px, "models": {}}
    curves = {}

    def run(model, files, transform=None, sources=None, min_px=min_face_px):
        return predict(model, tdata.eval_dataset(files, EVAL_BATCH, min_px, sources, transform))

    def decode(a):
        return np.char.decode(a.astype(bytes))

    for name, model in models.items():
        r = {}
        clean = run(model, test_files)
        y, p, g = clean["label"].astype(int), clean["prob"], decode(clean["group"])
        r["overall"] = summary(y, p)
        r["overall"]["auc_ci95"] = bootstrap_auc_ci(y, p, g)
        bal = balanced_indices(y)
        r["overall_balanced"] = summary(y[bal], p[bal])
        r["reliability"] = reliability(y[bal], p[bal])
        curves[name] = (y, p)

        src = decode(clean["source"])
        r["per_generator"] = {}
        for s in FAKE_SOURCES:
            m = (src == "wiki") | (src == s)
            r["per_generator"][s] = {**summary(y[m], p[m]),
                                     "auc_ci95": bootstrap_auc_ci(y[m], p[m], g[m], n_boot=500)}

        r["robustness"] = {}
        for pname, fn in tdata.PERTURBATIONS.items():
            pr = run(model, test_files, fn)
            r["robustness"][pname] = summary(pr["label"].astype(int), pr["prob"])

        sd = run(model, sdfvd_files, min_px=0)
        sy, sp, sg = sd["label"].astype(int), sd["prob"], decode(sd["group"])
        r["sdfvd_frame"] = {**summary(sy, sp), "auc_ci95": bootstrap_auc_ci(sy, sp, sg, n_boot=500)}
        videos = {}
        for n, prob, lab, grp in zip(decode(sd["name"]), sp, sy, sg):
            videos.setdefault(n.split(":")[0], {"p": [], "y": lab, "g": grp})["p"].append(prob)
        vy = np.array([v["y"] for v in videos.values()])
        vp = np.array([np.mean(v["p"]) for v in videos.values()])
        vg = np.array([v["g"] for v in videos.values()])
        r["sdfvd_video"] = {**summary(vy, vp), "auc_ci95": bootstrap_auc_ci(vy, vp, vg, n_boot=500)}
        results["models"][name] = r
        log(f"[evaluate:{name}] test AUC {r['overall']['auc']:.4f} {r['overall']['auc_ci95']}, "
            f"SDFVD video AUC {r['sdfvd_video']['auc']:.4f}")

    results["leave_one_generator_out"] = {}
    for s in FAKE_SOURCES:
        path = f"{out}/models/logo-{s}/model.keras"
        if not os.path.exists(path):
            continue
        held_out = keras.saving.load_model(path, compile=False)
        entry = {}
        for tag, model in [("held_out_model", held_out), ("full_model", models["ethicura"])]:
            pr = run(model, test_files, sources=["wiki", s])
            y, p, g = pr["label"].astype(int), pr["prob"], decode(pr["group"])
            entry[tag] = {**summary(y, p), "auc_ci95": bootstrap_auc_ci(y, p, g, n_boot=500)}
        pr = run(held_out, test_files, sources=["wiki"] + [x for x in FAKE_SOURCES if x != s])
        entry["held_out_model_on_seen_generators"] = summary(pr["label"].astype(int), pr["prob"])
        results["leave_one_generator_out"][s] = entry
        log(f"[evaluate:logo-{s}] unseen-generator AUC {entry['held_out_model']['auc']:.4f} "
            f"(trained-on-it {entry['full_model']['auc']:.4f})")

    with open(f"{out}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    render_figures(results, curves, out)
    log(f"[evaluate] wrote {out}/results.json and figures")


def render_figures(results, curves, out: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    colors = {"ethicura": "#d9534f", "baseline": "#6b7280"}
    names = {"ethicura": "Ethicura (this work)", "baseline": "Third-party baseline"}

    def save(fig, fname):
        fig.savefig(f"{out}/{fname}", dpi=160, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=colors[name], label=f"{names[name]}  AUC {results['models'][name]['overall']['auc']:.3f}")
    ax.plot([1e-3, 1], [1e-3, 1], ":", color="#bbb")
    ax.set(xlabel="False positive rate (real faces flagged)", ylabel="True positive rate (fakes caught)",
           title="ROC on held-out people", xscale="log", xlim=(1e-3, 1))
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    save(fig, "roc.png")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ":", color="#bbb", label="perfectly calibrated")
    for name, r in results["models"].items():
        pts = r["reliability"]
        ax.plot([a for a, _, _ in pts], [b for _, b, _ in pts], "o-", color=colors[name],
                label=f"{names[name]}  ECE {r['overall_balanced']['ece']:.3f}")
    ax.set(xlabel="Predicted P(fake)", ylabel="Observed fraction fake", title="Calibration (balanced test set)")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    save(fig, "calibration.png")

    perts = list(next(iter(results["models"].values()))["robustness"])
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (name, r) in enumerate(results["models"].items()):
        ax.bar([j + (i - 0.5) * 0.38 for j in range(len(perts))], [r["robustness"][k]["auc"] for k in perts],
               0.38, color=colors[name], label=names[name])
    ax.set_xticks(range(len(perts)), perts, rotation=30, ha="right")
    ax.set(ylabel="AUC", ylim=(0.4, 1.0), title="Robustness: AUC under degradation")
    ax.axhline(0.5, color="#bbb", ls=":"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    save(fig, "robustness.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train/evaluate Ethicura's detector.")
    parser.add_argument("step", choices=["download", "preprocess", "train", "calibrate", "evaluate", "all"])
    parser.add_argument("--data", default="/tmp/ethicura", help="downloaded datasets")
    parser.add_argument("--crops", help="face-crop TFRecords (default: <data>/crops)")
    parser.add_argument("--out", default="results", help="models, metrics, figures")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    parser.add_argument("--limit", type=int, default=0, help="smoke test: first N DFF pairs only")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    crops = args.crops or f"{args.data}/crops"

    if args.step in ("download", "all"):
        download(args.data)
    if args.step in ("preprocess", "all"):
        preprocess(args.data, crops, args.out, args.workers, args.limit)
    if args.step in ("train", "all"):
        for run_name in args.runs:
            train(crops, args.out, run_name, args.epochs, batch_size=args.batch_size)
    if args.step == "calibrate":
        for run_name in args.runs:
            calibrate(crops, args.out, run_name)
    if args.step in ("evaluate", "all"):
        evaluate(crops, args.out)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
