"""
ethicura_ai/src/gradcam.py
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.

Grad-CAM answers: "Which pixels in the face most influenced the fake/real decision?"
This is the interpretability layer that makes Ethicura AI transparent and auditable.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (ICCV 2017).
"""

import numpy as np
import tensorflow as tf


class GradCAM:
    """
    Computes Grad-CAM saliency maps for a given model and convolutional layer.

    For XceptionNet, we target 'block14_sepconv2_act' — the last spatial
    feature map before global average pooling. This gives the highest-resolution
    heatmap while still capturing high-level semantic features (facial artefacts).

    Args:
        model      : Trained Keras model (XceptionNet)
        layer_name : Name of the target convolutional layer
    """

    def __init__(self, model: tf.keras.Model, layer_name: str):
        self.model = model
        self.layer_name = layer_name
        self.target_layer = model.get_layer(layer_name)

        # NOTE on implementation: we deliberately do NOT build a spliced
        # tf.keras.Model like `Model(inputs=model.inputs, outputs=[target_layer.output, model.output])`.
        #
        # That approach works fine when layer_name is a plain conv layer
        # inside a flat model (the original Xception design), but breaks when
        # layer_name is a *nested Functional submodel* (our current
        # EfficientNetB0 setup, where the backbone is one named submodel
        # rather than flat conv layers). In Keras 3, reaching into a nested
        # submodel's output tensor this way raises
        # "ValueError: Output with path `0` is not connected to `inputs`"
        # and even when it doesn't raise, gradients w.r.t. that tensor come
        # back as None — the tape loses the connection across the submodel
        # boundary.
        #
        # The fix used in compute() below: replay the model's top-level
        # layers one at a time as plain callables *inside* the
        # GradientTape context, rather than asking Keras to build a new
        # graph object that reuses an existing internal tensor. This keeps
        # everything tape-tracked in a single eager pass and works for both
        # flat layers and nested submodels.
        self._layer_sequence = self._get_layers_after(layer_name)

        # Trace the forward+backward pass into a TF graph once, instead of
        # dispatching ~240 EfficientNet layers eagerly from Python on every
        # call. Pinned to a single-image signature so it never retraces.
        input_shape = (1,) + tuple(model.input_shape[1:])
        self._compute_graph = tf.function(
            self._compute_tensors,
            input_signature=[tf.TensorSpec(input_shape, tf.float32)],
        )

    def _get_layers_after(self, layer_name: str) -> list:
        """Return the model's layers in order, starting after (not including)
        the target layer, so compute() can replay the remaining forward pass."""
        layers = self.model.layers
        idx = next(i for i, l in enumerate(layers) if l.name == layer_name)
        return layers[idx + 1:]

    def compute(self, preprocessed_input: np.ndarray) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single preprocessed input.

        Args:
            preprocessed_input: np.ndarray of shape (1, H, W, 3)

        Returns:
            cam: np.ndarray of shape (h, w) with values in [0, 1]
                 where higher values = more influential for the FAKE prediction
        """
        cam, _ = self.compute_with_prediction(preprocessed_input)
        return cam

    def compute_with_prediction(self, preprocessed_input: np.ndarray):
        """
        Grad-CAM heatmap plus the model's fake probability, from one pass.

        The replayed forward pass in _compute_tensors runs the whole model
        (target submodel → pooling → dense), so its output *is* the
        prediction — no need for a separate model.predict() call.

        Returns:
            (cam, fake_prob): cam as in compute(), fake_prob as a float 0-1
        """
        cam, predictions = self._compute_graph(
            tf.convert_to_tensor(preprocessed_input, tf.float32)
        )
        return cam.numpy(), float(predictions[0, 0])

    def _compute_tensors(self, inputs):
        with tf.GradientTape() as tape:
            # Forward pass up to and including the target layer
            conv_outputs = self.target_layer(inputs, training=False)
            tape.watch(conv_outputs)

            # Replay the remaining layers manually (GlobalAveragePooling2D,
            # Dropout, Dense, etc.) so the tape stays connected across the
            # nested-submodel boundary. See __init__ for why this is
            # necessary instead of a spliced tf.keras.Model.
            x = conv_outputs
            for layer in self._layer_sequence:
                x = layer(x, training=False)
            predictions = x

            # For binary classification, index 0 = fake probability
            loss = predictions[:, 0]

        # Gradients of fake probability w.r.t. the target conv layer
        grads = tape.gradient(loss, conv_outputs)

        # Global average pooling over spatial dimensions → importance weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the feature maps by their importance
        conv_outputs = conv_outputs[0]
        cam = conv_outputs @ pooled_grads[..., tf.newaxis]
        cam = tf.squeeze(cam)

        # ReLU (only positive influences), then normalise to [0, 1]
        cam = tf.maximum(cam, 0) / (tf.math.reduce_max(cam) + 1e-8)

        return cam, predictions

    def compute_and_explain(
        self,
        preprocessed_input: np.ndarray,
        original_face_rgb: np.ndarray
    ) -> dict:
        """
        Full Grad-CAM explanation: heatmap + top-region text description.

        Returns:
            dict with:
                heatmap        : raw CAM array
                overlay        : coloured heatmap blended onto face
                hotspot_region : human-readable description of the most activated area
        """
        import cv2

        cam = self.compute(preprocessed_input)
        h, w = original_face_rgb.shape[:2]

        # Resize heatmap to face dimensions
        heatmap = cv2.resize(cam, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap)
        coloured = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        coloured_rgb = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original_face_rgb, 0.6, coloured_rgb, 0.4, 0)

        # Describe where the hotspot is (divide face into 3x3 grid)
        hotspot_region = self._describe_hotspot(heatmap)

        return {
            "heatmap": heatmap,
            "overlay": overlay,
            "hotspot_region": hotspot_region,
        }

    @staticmethod
    def _describe_hotspot(heatmap: np.ndarray) -> str:
        """Map peak activation location to a human-readable facial region."""
        h, w = heatmap.shape
        y_peak, x_peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)

        # Vertical region
        if y_peak < h / 3:
            v = "forehead/hairline"
        elif y_peak < 2 * h / 3:
            v = "mid-face (eyes/nose)"
        else:
            v = "lower face (mouth/chin)"

        # Horizontal region
        if x_peak < w / 3:
            h_desc = "left side"
        elif x_peak < 2 * w / 3:
            h_desc = "centre"
        else:
            h_desc = "right side"

        return f"{v}, {h_desc}"
