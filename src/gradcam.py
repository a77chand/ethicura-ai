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

        # Build a sub-model that outputs (target layer activations, final predictions)
        target_layer = model.get_layer(layer_name)
        self.grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[target_layer.output, model.output]
        )

    def compute(self, preprocessed_input: np.ndarray) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single preprocessed input.

        Args:
            preprocessed_input: np.ndarray of shape (1, 299, 299, 3)

        Returns:
            cam: np.ndarray of shape (H, W) with values in [0, 1]
                 where higher values = more influential for the FAKE prediction
        """
        with tf.GradientTape() as tape:
            inputs = tf.cast(preprocessed_input, tf.float32)
            conv_outputs, predictions = self.grad_model(inputs)
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

        return cam.numpy()

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
