"""open_clip and Hugging Face Transformers must give the same CLIP features,
so the probe's results don't depend on which backend loaded the weights.
Skipped unless the probe's optional dependencies are installed."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("open_clip")
pytest.importorskip("transformers")


def test_open_clip_and_transformers_embeddings_match():
    from probes.clip.run_probe import load_encoder

    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:224, 0:224]
    images = [np.clip(np.stack([xx, yy, (xx + yy) // 2], -1) * 0.8 + rng.integers(0, 60, (224, 224, 3)), 0, 255)
              .astype(np.uint8) for _ in range(4)]
    a = load_encoder("open_clip", "ViT-B-16", "openai", "cpu")(images)
    b = load_encoder("hf", "ViT-B-16", "openai", "cpu")(images)
    cosine = (a * b).sum(axis=1)
    assert a.shape == b.shape == (4, 512)
    assert cosine.min() > 0.999, cosine
