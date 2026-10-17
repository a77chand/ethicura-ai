"""
Export a detector to ONNX for in-browser inference (onnxruntime-web).

    python -m training.export_web --out ../ethicura-frontend/public/models/detector.onnx
    python -m training.export_web --model path/to/model.keras --out ...

The exported graph takes a 1x224x224x3 float32 face crop in [0, 1] and
returns:
    prob : (1, 1)  P(fake)
    cam  : (1, 7, 7) heatmap in [0, 1]

The heatmap is computed in-graph without a backward pass. For a head of
global-average-pool → dropout → dense(sigmoid), Grad-CAM's channel weights
are mean_ij dσ/dA_k(i,j) = σ'(z)·w_k / (H·W), a positive scalar times the
dense weight w_k. After ReLU and max-normalisation the scalar cancels, so
Grad-CAM equals classic CAM: ReLU(Σ_k w_k A_k) / max. The export checks
this numerically against src/gradcam.py before writing anything.
"""

import argparse
import os

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")

import numpy as np
import tensorflow as tf

BASELINE_REPO = "ariyul/deepfake-detection-cnn_v2"


def load(model_path: str | None):
    import keras

    if model_path:
        return keras.saving.load_model(model_path, compile=False)
    from huggingface_hub import snapshot_download

    path = snapshot_download(BASELINE_REPO)
    return keras.saving.load_model(f"{path}/keras_model/best_model.keras", compile=False)


def cam_function(model, backbone_name: str = "efficientnetb0"):
    """tf.function: face (1,224,224,3) → (prob (1,1), cam (1,h,w))."""
    import keras

    layers = model.layers
    idx = next(i for i, l in enumerate(layers) if l.name == backbone_name)
    before = [l for l in layers[:idx] if not isinstance(l, keras.layers.InputLayer)]
    backbone, after = layers[idx], layers[idx + 1:]
    dense = after[-1]
    kernel = tf.constant(dense.get_weights()[0][:, 0], tf.float32)  # (channels,)

    @tf.function(input_signature=[tf.TensorSpec((1, 224, 224, 3), tf.float32, name="face")])
    def run(face):
        x = face
        for layer in before:
            x = layer(x, training=False)
        features = backbone(x, training=False)
        y = features
        for layer in after:
            y = layer(y, training=False)
        cam = tf.nn.relu(tf.tensordot(features, kernel, axes=[[3], [0]]))
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        return {"prob": tf.identity(y, name="prob"), "cam": tf.identity(cam, name="cam")}

    return run


def verify(model, run, n: int = 4):
    """CAM path must reproduce the model's probability and src.gradcam's heatmap."""
    from src.gradcam import GradCAM

    gradcam = GradCAM(model, "efficientnetb0")
    rng = np.random.default_rng(0)
    for _ in range(n):
        x = rng.random((1, 224, 224, 3), dtype=np.float32)
        out = run(tf.constant(x))
        ref_cam, ref_prob = gradcam.compute_with_prediction(x)
        assert abs(float(out["prob"][0, 0]) - ref_prob) < 1e-4, "probability mismatch"
        np.testing.assert_allclose(out["cam"].numpy()[0], ref_cam, atol=1e-3)


def rename_io(path: str, inputs: list[str], outputs: list[str]):
    """Give the ONNX graph stable input/output names ('face', 'prob', 'cam')."""
    import onnx

    m = onnx.load(path)
    mapping = {}
    for value, name in zip(list(m.graph.input), inputs):
        mapping[value.name] = name
    for value, name in zip(list(m.graph.output), outputs):
        mapping[value.name] = name
    for node in m.graph.node:
        node.input[:] = [mapping.get(n, n) for n in node.input]
        node.output[:] = [mapping.get(n, n) for n in node.output]
    for value in list(m.graph.input) + list(m.graph.output):
        value.name = mapping.get(value.name, value.name)
    onnx.save(m, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", help="path to a .keras model (default: current baseline from the Hub)")
    parser.add_argument("--out", required=True, help="output .onnx path")
    args = parser.parse_args()

    import onnxruntime as ort
    import tf2onnx

    model = load(args.model)
    run = cam_function(model)
    verify(model, run)
    print("CAM == Grad-CAM and probabilities match on random inputs")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # Freeze weights and captured constants into the graph first; converting
    # the tf.function directly leaves some captures as dangling ONNX inputs.
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    frozen = convert_variables_to_constants_v2(run.get_concrete_function())
    graph_def = frozen.graph.as_graph_def()
    tf2onnx.convert.from_graph_def(
        graph_def, opset=17, output_path=args.out,
        input_names=[t.name for t in frozen.inputs],
        output_names=[t.name for t in frozen.outputs])
    # tf.nest flattens the {"prob", "cam"} dict in sorted key order.
    rename_io(args.out, ["face"], sorted(run.get_concrete_function().structured_outputs))

    # The ONNX model must agree with TensorFlow.
    session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(1).random((1, 224, 224, 3), dtype=np.float32)
    ref = run(tf.constant(x))
    got = dict(zip([o.name for o in session.get_outputs()], session.run(None, {"face": x})))
    assert abs(float(got["prob"][0, 0]) - float(ref["prob"][0, 0])) < 1e-4
    np.testing.assert_allclose(got["cam"], ref["cam"].numpy(), atol=1e-3)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out} ({size_mb:.1f} MB); ONNX output matches TensorFlow")


if __name__ == "__main__":
    main()
