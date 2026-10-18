"""
Render evaluation results as Markdown for the README, straight from the
files the training run wrote — numbers are never typed by hand.

    python -m training.report results > results/report.md
"""

import json
import os
import sys

GENERATORS = {"inpainting": "SD inpainting", "insight": "InsightFace swap", "text2img": "SD text-to-image"}
PERTURBATIONS = {
    "clean": "clean", "jpeg_q90": "JPEG q90", "jpeg_q70": "JPEG q70", "jpeg_q50": "JPEG q50",
    "jpeg_q30": "JPEG q30", "downscale_0.5": "downscale ×0.5", "downscale_0.25": "downscale ×0.25",
    "blur_s1": "blur σ=1", "blur_s2": "blur σ=2",
}


def ci(metric: dict) -> str:
    lo, hi = metric["auc_ci95"]
    return f"{metric['auc']:.3f} ({lo:.3f}–{hi:.3f})"


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def row(label: str, ours: float, base: float, fmt, lower_is_better: bool = False,
        ours_text: str | None = None, base_text: str | None = None) -> str:
    """Table row with the better model in bold (ties: neither)."""
    a, b = ours_text or fmt(ours), base_text or fmt(base)
    if ours != base:
        if (ours < base) == lower_is_better:
            a = f"**{a}**"
        else:
            b = f"**{b}**"
    return f"| {label} | {a} | {b} |"


def calibration_text(meta: dict) -> str:
    c = meta.get("calibration")
    if c:
        return f"Platt scaling (a = {c['a']:.2f}, b = {c['b']:.2f} on the raw logits)"
    return f"temperature scaling (T = {meta.get('temperature', 1.0):.2f})"


def render(out_dir: str) -> str:
    with open(f"{out_dir}/results.json") as f:
        r = json.load(f)
    ours, base = r["models"]["ethicura"], r["models"]["baseline"]
    meta = {}
    models_dir = f"{out_dir}/models"
    for run in sorted(os.listdir(models_dir)) if os.path.isdir(models_dir) else []:
        path = f"{models_dir}/{run}/meta.json"
        if os.path.exists(path):
            with open(path) as f:
                meta[run] = json.load(f)
    lines = []
    o, b = ours["overall"], base["overall"]
    lines += [
        f"Test set: **{o['n_real']:,} real and {o['n_fake']:,} fake faces** from people never seen in training "
        f"(faces ≥ {r['min_face_px']} px). 95% CIs from a person-level bootstrap.",
        "",
        "| Metric | Ethicura (this work) | Third-party baseline |",
        "|---|---|---|",
        row("AUC (95% CI)", o["auc"], b["auc"], None, ours_text=ci(o), base_text=ci(b)),
        row("Equal error rate", o["eer"], b["eer"], pct, lower_is_better=True),
        row("Fakes caught at 1% false alarms", o["tpr_at_fpr_1pct"], b["tpr_at_fpr_1pct"], pct),
        row("Fakes caught at 5% false alarms", o["tpr_at_fpr_5pct"], b["tpr_at_fpr_5pct"], pct),
        row("Balanced accuracy @ 0.5", o["balanced_accuracy"], b["balanced_accuracy"], pct),
        row("Calibration error (ECE, balanced)", ours["overall_balanced"]["ece"], base["overall_balanced"]["ece"],
            lambda x: f"{x:.3f}", lower_is_better=True),
        "",
        "**Per generator** (real test faces vs. one generator's fakes):",
        "",
        "| Generator | Ethicura AUC | Baseline AUC |",
        "|---|---|---|",
    ]
    for g, label in GENERATORS.items():
        og, bg = ours["per_generator"][g], base["per_generator"][g]
        lines.append(row(label, og["auc"], bg["auc"], None, ours_text=ci(og), base_text=ci(bg)))

    logo = r.get("leave_one_generator_out", {})
    if logo:
        lines += [
            "",
            "**Generalisation to an unseen generator.** Each row is a model trained *without* that "
            "generator, tested on it, next to the full model that did see it in training:",
            "",
            "| Held-out generator | AUC, never seen | AUC, seen in training | Same held-out model on the generators it saw |",
            "|---|---|---|---|",
        ]
        for g, e in logo.items():
            lines.append(f"| {GENERATORS[g]} | {ci(e['held_out_model'])} | {ci(e['full_model'])} | "
                         f"{e['held_out_model_on_seen_generators']['auc']:.3f} |")

    lines += [
        "",
        "**Robustness** (AUC with the degradation applied to every test face, real and fake):",
        "",
        "| Condition | " + " | ".join(PERTURBATIONS[k] for k in ours["robustness"]) + " |",
        "|---|" + "---|" * len(ours["robustness"]),
        "| Ethicura | " + " | ".join(f"{v['auc']:.3f}" for v in ours["robustness"].values()) + " |",
        "| Baseline | " + " | ".join(f"{v['auc']:.3f}" for v in base["robustness"].values()) + " |",
        "",
        "**Out of domain: SDFVD face-swap videos** (a different dataset and manipulation tool, never "
        "used for training or model selection):",
        "",
        "| | Ethicura | Baseline |",
        "|---|---|---|",
        row("Frame-level AUC", ours["sdfvd_frame"]["auc"], base["sdfvd_frame"]["auc"], None,
            ours_text=ci(ours["sdfvd_frame"]), base_text=ci(base["sdfvd_frame"])),
        row(f"Video-level AUC (mean over frames, {ours['sdfvd_video']['n']} videos)",
            ours["sdfvd_video"]["auc"], base["sdfvd_video"]["auc"], None,
            ours_text=ci(ours["sdfvd_video"]), base_text=ci(base["sdfvd_video"])),
    ]
    if "main" in meta:
        m = meta["main"]
        cal = m["val_balanced"]
        lines += [
            "",
            f"Training: {m['train_reals']:,} real faces per epoch against the same number of fakes, "
            f"best epoch {m['best_epoch']} of {m['config']['epochs']}, {m['train_minutes']} min on "
            f"{'a GPU' if m['config'].get('mixed_precision') else 'CPU'}. Calibration: {calibration_text(m)} "
            f"fitted on balanced validation (ECE {cal['calibrated']['ece']:.3f}) and folded into the final layer.",
        ]
    return "\n".join(lines) + "\n"


def render_clip(out_dir: str) -> str:
    """Ethicura vs the PyTorch CLIP linear probe (probes/clip), same test set."""
    with open(f"{out_dir}/results.json") as f:
        r = json.load(f)
    with open(f"{out_dir}/clip_probe.json") as f:
        c = json.load(f)
    e = r["models"]["ethicura"]
    logo = r["leave_one_generator_out"]
    rows = [
        row("AUC (95% CI)", e["overall"]["auc"], c["overall"]["auc"], None,
            ours_text=ci(e["overall"]), base_text=ci(c["overall"])),
        row("Fakes caught at 1% false alarms", e["overall"]["tpr_at_fpr_1pct"], c["overall"]["tpr_at_fpr_1pct"], pct),
        row("Balanced accuracy @ 0.5", e["overall_balanced"]["balanced_accuracy"],
            c["overall_balanced"]["balanced_accuracy"], pct),
    ]
    for g, label in GENERATORS.items():
        rows.append(row(f"Never saw {label}", logo[g]["held_out_model"]["auc"],
                        c["leave_one_generator_out"][g]["auc"], None,
                        ours_text=ci(logo[g]["held_out_model"]), base_text=ci(c["leave_one_generator_out"][g])))
    rows.append(row("SDFVD video-level AUC (out of domain)", e["sdfvd_video"]["auc"], c["sdfvd_video"]["auc"], None,
                    ours_text=ci(e["sdfvd_video"]), base_text=ci(c["sdfvd_video"])))
    return "\n".join([
        f"| | Ethicura EfficientNetB0 (fine-tuned) | {c['encoder']} + linear probe |",
        "|---|---|---|", *rows]) + "\n"


def splice(readme_path: str, markdown: str, tag: str = "results"):
    """Replace the README section between <!-- {tag}:start --> and <!-- {tag}:end -->."""
    start, end = f"<!-- {tag}:start -->", f"<!-- {tag}:end -->"
    with open(readme_path) as f:
        text = f.read()
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    with open(readme_path, "w") as f:
        f.write(f"{head}{start}\n{markdown}{end}{tail}")


if __name__ == "__main__":
    # python -m training.report results [--readme README.md]
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    markdown = render(out_dir)
    if "--readme" in sys.argv:
        readme = sys.argv[sys.argv.index("--readme") + 1]
        splice(readme, markdown)
        if os.path.exists(f"{out_dir}/clip_probe.json"):
            splice(readme, render_clip(out_dir), "clip")
    else:
        print(markdown)
