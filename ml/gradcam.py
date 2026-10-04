from __future__ import annotations

import io
import base64

import numpy as np


def generate_gradcam(
    model,
    image_array: np.ndarray,
    predicted_class: int,
    last_conv_layer_name: str | None = None,
) -> np.ndarray:
    import tensorflow as tf

    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if hasattr(layer, "layers"):
                for sub_layer in reversed(layer.layers):
                    if len(sub_layer.output_shape) == 4:
                        last_conv_layer_name = sub_layer.name
                        break
                if last_conv_layer_name:
                    break
            elif len(layer.output_shape) == 4:
                last_conv_layer_name = layer.name
                break

    base_model = None
    for layer in model.layers:
        if hasattr(layer, "layers"):
            try:
                layer.get_layer(last_conv_layer_name)
                base_model = layer
                break
            except ValueError:
                continue

    if base_model is not None:
        conv_output = base_model.get_layer(last_conv_layer_name).output
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.output, conv_output],
        )
    else:
        conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[model.output, conv_layer.output],
        )

    img_tensor = tf.cast(image_array[np.newaxis, ...], tf.float32)
    with tf.GradientTape() as tape:
        predictions, conv_outputs = grad_model(img_tensor, training=False)
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    import cv2

    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(original_image * (1 - alpha) + colormap * alpha)
    return overlay


def gradcam_to_base64(
    model,
    image_array: np.ndarray,
    predicted_class: int,
) -> str:
    from PIL import Image

    heatmap = generate_gradcam(model, image_array, predicted_class)
    overlay = overlay_heatmap(image_array, heatmap)
    img = Image.fromarray(overlay)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
