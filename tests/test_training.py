"""Unit tests for the training utilities: model contract, calibration, metrics, data pipeline."""

import numpy as np
import pytest
import tensorflow as tf

from training import data, metrics
from training.model import build_model, fit_temperature, fold_temperature, logit


@pytest.fixture(scope="module")
def model():
    m, _ = build_model(weights=None)
    return m


def test_model_contract(model):
    """224x224x3 in [0,1] → one probability; backbone reachable for Grad-CAM."""
    out = model(np.random.rand(2, 224, 224, 3).astype("float32"), training=False)
    assert out.shape == (2, 1)
    assert np.all((out.numpy() >= 0) & (out.numpy() <= 1))
    assert model.get_layer("efficientnetb0") is not None


def test_gradcam_replays_layers_before_backbone(model):
    """The Rescaling layer sits before the backbone; Grad-CAM must run it, or
    its prediction would differ from the model's own."""
    from src.gradcam import GradCAM

    x = np.random.rand(1, 224, 224, 3).astype("float32")
    cam, prob = GradCAM(model, "efficientnetb0").compute_with_prediction(x)
    assert cam.shape == (7, 7)
    assert prob == pytest.approx(float(model(x, training=False)[0, 0]), abs=1e-5)


def test_fold_temperature_matches_scaled_logits(model):
    x = np.random.rand(4, 224, 224, 3).astype("float32")
    before = model(x, training=False).numpy()[:, 0]
    fold_temperature(model, 2.5)
    after = model(x, training=False).numpy()[:, 0]
    expected = 1 / (1 + np.exp(-logit(before) / 2.5))
    np.testing.assert_allclose(after, expected, atol=1e-5)
    fold_temperature(model, 1 / 2.5)  # undo for other tests


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 2, 20000)
    labels = rng.random(20000) < 1 / (1 + np.exp(-true_logits))
    overconfident = 1 / (1 + np.exp(-3 * true_logits))
    assert fit_temperature(overconfident, labels) == pytest.approx(3.0, rel=0.05)


def test_fold_platt_matches_affine_logits(model):
    from training.model import fold_platt

    x = np.random.rand(4, 224, 224, 3).astype("float32")
    before = model(x, training=False).numpy()[:, 0]
    fold_platt(model, 2.0, 1.5)
    after = model(x, training=False).numpy()[:, 0]
    np.testing.assert_allclose(after, 1 / (1 + np.exp(-(2.0 * logit(before) + 1.5))), atol=1e-5)
    fold_platt(model, 0.5, -0.75)  # undo: 0.5·(2z + 1.5) − 0.75 = z


def test_fit_platt_recovers_scale_and_shift():
    from training.model import fit_platt

    rng = np.random.default_rng(3)
    true_logits = rng.normal(0, 2, 40000)
    labels = rng.random(40000) < 1 / (1 + np.exp(-true_logits))
    # A model whose logits are (true - 1.0) / 2: under-confident and shifted.
    reported = 1 / (1 + np.exp(-(true_logits - 1.0) / 2))
    a, b = fit_platt(reported, labels)
    assert a == pytest.approx(2.0, rel=0.05) and b == pytest.approx(1.0, abs=0.08)


def test_summary_metrics_perfect_and_random():
    y = np.array([0] * 50 + [1] * 50)
    perfect = metrics.summary(y, y * 0.9 + 0.05)
    assert perfect["auc"] == 1.0 and perfect["eer"] == 0.0 and perfect["accuracy"] == 1.0
    rng = np.random.default_rng(1)
    random = metrics.summary(np.repeat([0, 1], 5000), rng.random(10000))
    assert random["auc"] == pytest.approx(0.5, abs=0.03)


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.3 + rng.random(2000) * 0.7, 0, 1)
    groups = rng.integers(0, 200, 2000)
    lo, hi = metrics.bootstrap_auc_ci(y, p, groups, n_boot=200)
    point = metrics.summary(y, p)["auc"]
    assert lo < point < hi


def test_ece_zero_when_calibrated():
    probs = np.repeat([0.2, 0.8], 5000)
    labels = np.concatenate([np.r_[np.ones(1000), np.zeros(4000)], np.r_[np.ones(4000), np.zeros(1000)]])
    assert metrics.ece(labels, probs) == pytest.approx(0.0, abs=1e-9)


def test_augment_and_perturbations_keep_shape_and_range():
    image = tf.random.uniform([224, 224, 3])
    for fn in [data.augment] + [f for f in data.PERTURBATIONS.values() if f is not None]:
        out = fn(image)
        assert out.shape == (224, 224, 3)
        assert float(tf.reduce_min(out)) >= 0.0 and float(tf.reduce_max(out)) <= 1.0


def _write_records(path, items):
    import cv2

    with tf.io.TFRecordWriter(str(path)) as w:
        for i, (source, label) in enumerate(items):
            img = np.full((224, 224, 3), 40 * label + 10, np.uint8)
            jpg = cv2.imencode(".jpg", img)[1].tobytes()
            ex = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[jpg])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                "source": tf.train.Feature(bytes_list=tf.train.BytesList(value=[source.encode()])),
                "group": tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(i % 7).encode()])),
                "name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[f"img{i}".encode()])),
                "face_px": tf.train.Feature(int64_list=tf.train.Int64List(value=[50 + i])),
            }))
            w.write(ex.SerializeToString())


def test_training_stream_is_class_balanced(tmp_path):
    path = tmp_path / "shard.tfrecord"
    # 3x more fakes than reals on disk — the stream should still be ~50/50.
    _write_records(path, [("wiki", 0)] * 20 + [("inpainting", 1)] * 20 + [("insight", 1)] * 20 + [("text2img", 1)] * 20)
    ds = data.training_dataset([str(path)], batch_size=50, fake_sources=["inpainting", "insight", "text2img"])
    labels = np.concatenate([y.numpy() for _, y in ds.take(20)])
    assert labels.mean() == pytest.approx(0.5, abs=0.05)


def test_eval_dataset_filters(tmp_path):
    path = tmp_path / "shard.tfrecord"
    _write_records(path, [("wiki", 0)] * 10 + [("insight", 1)] * 10)
    rows = list(data.eval_dataset([str(path)], 64, min_face_px=55, sources=["insight"]).unbatch())
    # face_px = 50 + i; insight items are i = 10..19, all ≥ 55
    assert len(rows) == 10
    assert all(r[2].numpy() == b"insight" for r in rows)
