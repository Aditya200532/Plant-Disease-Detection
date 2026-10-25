from __future__ import annotations

import io
import base64

import numpy as np


def find_last_conv_layer_name(model) -> str:
    import tensorflow as tf

    containers = [layer for layer in model.layers if getattr(layer, "layers", None)]
    search_order = list(reversed(containers)) + [model]
    for container in search_order:
        for sub_layer in reversed(container.layers):
            if isinstance(sub_layer, tf.keras.layers.Conv2D):
                return sub_layer.name
    raise ValueError("No Conv2D layer found in model; cannot build Grad-CAM.")


def build_grad_model(model, last_conv_layer_name: str | None = None):
    """Keras 3 safe Grad-CAM model.

    The trained model nests MobileNetV2 inside an outer Functional graph, so
    tensors from the inner blueprint cannot be mixed with model.input. Instead
    the inner model is re-wrapped with the conv output exposed and then called
    while rebuilding the outer graph, keeping every tensor in one graph.
    """
    import tensorflow as tf

    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer_name(model)

    container = None
    for layer in model.layers:
        subs = getattr(layer, "layers", None)
        if subs and any(sub.name == last_conv_layer_name for sub in subs):
            container = layer
            break

    if container is None:
        conv_output = model.get_layer(last_conv_layer_name).output
        return tf.keras.Model(model.input, [model.output, conv_output])

    inner = container
    inner_grad = tf.keras.Model(
        inputs=inner.input,
        outputs=[inner.output, inner.get_layer(last_conv_layer_name).output],
    )

    inputs = tf.keras.Input(shape=tuple(model.input.shape[1:]))
    x = inputs
    conv_output = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if layer is container:
            x, conv_output = inner_grad(x)
        else:
            x = layer(x)
    return tf.keras.Model(inputs, [x, conv_output])


def generate_gradcam(
    grad_model,
    image_array: np.ndarray,
    predicted_class: int,
) -> np.ndarray:
    import tensorflow as tf

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
    from PIL import Image
    import matplotlib.cm as cm

    height, width = original_image.shape[:2]
    resized = Image.fromarray(np.uint8(255 * heatmap)).resize(
        (width, height), Image.BILINEAR
    )
    heat = np.asarray(resized, dtype=np.float32) / 255.0
    colormap = np.asarray(cm.jet(heat)[..., :3], dtype=np.float32) * 255.0
    overlay = original_image * (1 - alpha) + colormap * alpha
    return np.uint8(np.clip(overlay, 0, 255))


def gradcam_to_base64(
    grad_model,
    image_array: np.ndarray,
    predicted_class: int,
) -> str:
    from PIL import Image

    heatmap = generate_gradcam(grad_model, image_array, predicted_class)
    overlay = overlay_heatmap(image_array, heatmap)
    img = Image.fromarray(overlay)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
