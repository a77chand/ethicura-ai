"""
PyTorch baseline: a linear probe on frozen CLIP ViT-L/14 features (the UnivFD
recipe, Ojha et al., CVPR 2023), evaluated exactly like Ethicura's detector.

    pip install -r probes/clip/requirements.txt
    python -m probes.clip.run_probe --crops /path/to/crops --cache /tmp/clip_cache

Reads the face-crop TFRecords produced by training/run.py (read-only: same
images, splits and filters as the main evaluation), writes only to --cache
and results/clip_probe.json. Nothing else in the repo is touched.
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

from training.metrics import balanced_indices, bootstrap_auc_ci, summary
from training.model import fit_platt, logit

FAKES = ["inpainting", "insight", "text2img"]
MIN_FACE_PX = 80  # same filter as the main evaluation


# ── Stage 1: CLIP embeddings ──────────────────────────────────────────────────

def records(crops: str, split: str):
    """Yield (uint8 RGB image, label, source, group, name, face_px) from the crops."""
    import tensorflow as tf  # only for reading TFRecords

    tf.config.set_visible_devices([], "GPU")
    spec = {k: tf.io.FixedLenFeature([], t) for k, t in [
        ("image", tf.string), ("label", tf.int64), ("source", tf.string),
        ("group", tf.string), ("name", tf.string), ("face_px", tf.int64)]}
    files = sorted(glob.glob(f"{crops}/{split}/*.tfrecord"))
    for rec in tf.data.TFRecordDataset(files):
        ex = tf.io.parse_single_example(rec, spec)
        yield (tf.io.decode_jpeg(ex["image"], channels=3).numpy(), int(ex["label"]),
               ex["source"].numpy().decode(), ex["group"].numpy().decode(),
               ex["name"].numpy().decode(), int(ex["face_px"]))


@torch.no_grad()
def embed(crops: str, split: str, cache: str, model, preprocess, device, batch: int = 64):
    path = f"{cache}/{split}.npz"
    if os.path.exists(path):
        return dict(np.load(path))
    from PIL import Image

    feats, meta, buf = [], [], []

    def flush():
        x = torch.stack([preprocess(Image.fromarray(im)) for im in buf]).to(device)
        f = model.encode_image(x).float()
        feats.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
        buf.clear()

    for n, (img, label, source, group, name, face_px) in enumerate(records(crops, split), 1):
        if split != "sdfvd" and face_px < MIN_FACE_PX:
            continue
        buf.append(img)
        meta.append((label, source, group, name))
        if len(buf) == batch:
            flush()
        if n % 5000 == 0:
            print(f"[embed:{split}] {n} records", flush=True)
    if buf:
        flush()
    out = {"x": np.concatenate(feats).astype(np.float32),
           "y": np.array([m[0] for m in meta]), "source": np.array([m[1] for m in meta]),
           "group": np.array([m[2] for m in meta]), "name": np.array([m[3] for m in meta])}
    np.savez(path, **out)
    print(f"[embed:{split}] {len(out['y'])} faces → {path}", flush=True)
    return out


# ── Stage 2: linear probe ─────────────────────────────────────────────────────

def keep(d, sources):
    m = np.isin(d["source"], ["wiki"] + sources)
    return {k: v[m] for k, v in d.items()}


def train_probe(tr, va, seed: int = 0, epochs: int = 30):
    """nn.Linear on CLIP features. Loss weights make real vs fake 50/50 and the
    fakes equal across generators, like the main model's balanced batches.
    Best epoch by validation AUC; then Platt calibration on balanced val."""
    torch.manual_seed(seed)
    x, y = torch.tensor(tr["x"]), torch.tensor(tr["y"], dtype=torch.float32)
    w = np.ones(len(y), dtype=np.float32)
    w[tr["y"] == 0] = 0.5 / (tr["y"] == 0).sum()
    fakes = [s for s in np.unique(tr["source"]) if s != "wiki"]
    for s in fakes:
        w[tr["source"] == s] = 0.5 / len(fakes) / (tr["source"] == s).sum()
    w = torch.tensor(w * len(w))

    probe = torch.nn.Linear(x.shape[1], 1)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    xv = torch.tensor(va["x"])
    best, best_state = -1.0, None
    for epoch in range(epochs):
        perm = torch.randperm(len(y))
        for i in range(0, len(y), 512):
            idx = perm[i:i + 512]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                probe(x[idx]).squeeze(1), y[idx], weight=w[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            auc = summary(va["y"], torch.sigmoid(probe(xv)).squeeze(1).numpy())["auc"]
        if auc > best:
            best, best_state = auc, {k: v.clone() for k, v in probe.state_dict().items()}
    probe.load_state_dict(best_state)

    with torch.no_grad():
        pv = torch.sigmoid(probe(xv)).squeeze(1).numpy()
    bi = balanced_indices(va["y"])
    a, b = fit_platt(pv[bi], va["y"][bi])

    def predict(d):
        with torch.no_grad():
            p = torch.sigmoid(probe(torch.tensor(d["x"]))).squeeze(1).numpy()
        return 1 / (1 + np.exp(-(a * logit(p) + b)))

    return predict, best


def scored(y, p, g, n_boot=1000):
    return {**summary(y, p), "auc_ci95": bootstrap_auc_ci(y, p, g, n_boot=n_boot)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True)
    ap.add_argument("--cache", default="/tmp/clip_cache")
    ap.add_argument("--model", default="ViT-L-14")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--out", default="results/clip_probe.json")
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)

    import open_clip

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, device=device)
    model.eval()
    data = {s: embed(args.crops, s, args.cache, model, preprocess, device) for s in ["train", "val", "test", "sdfvd"]}
    tr, va, te, sd = data["train"], data["val"], data["test"], data["sdfvd"]

    results = {"encoder": f"CLIP {args.model} ({args.pretrained}), frozen", "probe": "torch.nn.Linear, AdamW",
               "train_faces": int(len(tr["y"]))}
    predict, val_auc = train_probe(tr, va)
    p = predict(te)
    bi = balanced_indices(te["y"])
    results["val_auc"] = val_auc
    results["overall"] = scored(te["y"], p, te["group"])
    results["overall_balanced"] = summary(te["y"][bi], p[bi])
    results["per_generator"] = {s: scored(te["y"][m], p[m], te["group"][m], 500)
                                for s in FAKES for m in [np.isin(te["source"], ["wiki", s])]}

    ps = predict(sd)
    results["sdfvd_frame"] = scored(sd["y"], ps, sd["group"], 500)
    vids = sorted({n.split(":")[0] for n in sd["name"]})
    vid = np.array([n.split(":")[0] for n in sd["name"]])
    vy = np.array([sd["y"][vid == v][0] for v in vids])
    vp = np.array([ps[vid == v].mean() for v in vids])
    vg = np.array([v.lstrip("vs") for v in vids])
    results["sdfvd_video"] = scored(vy, vp, vg, 500)

    results["leave_one_generator_out"] = {}
    for held in FAKES:
        seen = [s for s in FAKES if s != held]
        pred_h, _ = train_probe(keep(tr, seen), keep(va, seen))
        held_te = keep(te, [held])
        results["leave_one_generator_out"][held] = scored(held_te["y"], pred_h(held_te), held_te["group"], 500)
        print(f"[probe] never saw {held}: AUC {results['leave_one_generator_out'][held]['auc']:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    o = results["overall"]
    print(f"[probe] test AUC {o['auc']:.4f} {o['auc_ci95']}  TPR@1% {o['tpr_at_fpr_1pct']:.3f}  "
          f"bal-acc {results['overall_balanced']['balanced_accuracy']:.3f}  "
          f"SDFVD video {results['sdfvd_video']['auc']:.3f} → {args.out}")


if __name__ == "__main__":
    main()
