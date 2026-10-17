"""
Model definition and post-hoc calibration.

The architecture keeps the API's contract: input is a 224x224 RGB face crop
scaled to [0, 1], output is P(fake), and the backbone is a nested submodel
named "efficientnetb0" so src/gradcam.py works unchanged.
"""

import numpy as np

IMAGE_SIZE = 224


def build_model(dropout: float = 0.3, weights: str | None = "imagenet"):
    """Returns (model, backbone). EfficientNetB0 was chosen for the live
    service's CPU latency budget (~30 ms per face on 2 vCPU)."""
    import keras

    inputs = keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), name="face")
    # Keras' EfficientNet normalises 0-255 input internally; the API sends
    # [0, 1], so scale back up inside the model.
    x = keras.layers.Rescaling(255.0, name="to_0_255")(inputs)
    backbone = keras.applications.EfficientNetB0(
        include_top=False, weights=weights, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    # training=False keeps BatchNorm statistics frozen while fine-tuning.
    x = backbone(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="pool")(x)
    x = keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="fake_prob")(x)
    return keras.Model(inputs, outputs, name="ethicura_effnetb0"), backbone


def to_float32(model):
    """Rebuild a (possibly mixed-precision) trained model in float32 for CPU
    inference and copy the weights across."""
    import keras

    keras.mixed_precision.set_global_policy("float32")
    clean, _ = build_model(dropout=model.get_layer("dropout").rate, weights=None)
    clean.set_weights(model.get_weights())
    return clean


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)


def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    """Temperature T minimising validation NLL of sigmoid(logit / T)."""
    z = logit(probs)
    y = labels.astype(np.float64)

    def nll(t):
        q = 1.0 / (1.0 + np.exp(-z / t))
        q = np.clip(q, 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

    # Golden-section search over log T in [1/20, 20].
    lo, hi = np.log(0.05), np.log(20.0)
    g = (np.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(80):
        if nll(np.exp(a)) < nll(np.exp(b)):
            hi = b
        else:
            lo = a
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    return float(np.exp((lo + hi) / 2))


def fit_platt(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Platt scaling: (a, b) minimising NLL of sigmoid(a·logit + b).

    Temperature scaling (a = 1/T, b = 0) can only sharpen or soften scores;
    when a model is systematically under-confident on one class it also
    needs a shift, which b provides."""
    z = logit(probs)
    y = labels.astype(np.float64)
    a, b = 1.0, 0.0
    for _ in range(100):  # Newton's method; the NLL is convex in (a, b)
        q = 1.0 / (1.0 + np.exp(-(a * z + b)))
        grad = np.array([np.sum((q - y) * z), np.sum(q - y)])
        w = q * (1 - q)
        hess = np.array([[np.sum(w * z * z), np.sum(w * z)], [np.sum(w * z), np.sum(w)]])
        step = np.linalg.solve(hess + 1e-9 * np.eye(2), grad)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-10:
            break
    return float(a), float(b)


def fold_platt(model, a: float, b: float):
    """sigmoid(a·(w·x + c) + b) == sigmoid((a·w)·x + (a·c + b)): bake Platt
    scaling into the last layer, like fold_temperature."""
    dense = model.get_layer("fake_prob")
    kernel, bias = dense.get_weights()
    dense.set_weights([kernel * a, bias * a + b])
    return model


def fold_temperature(model, temperature: float):
    """sigmoid((w·x + b) / T) == sigmoid((w/T)·x + b/T): bake calibration
    into the last layer so serving needs no extra code."""
    dense = model.get_layer("fake_prob")
    kernel, bias = dense.get_weights()
    dense.set_weights([kernel / temperature, bias / temperature])
    return model
