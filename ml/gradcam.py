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
            x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
            x, conv_output = inner_grad(x, training=False)
        elif isinstance(layer, (tf.keras.Sequential, tf.keras.layers.Dropout)):
            x = layer(x, training=False)
        else:
            x = layer(x)
    return tf.keras.Model(inputs, [x, conv_output])


def predict_and_generate_gradcam(
    grad_model,
    image_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import tensorflow as tf

    img_tensor = tf.convert_to_tensor(image_array[np.newaxis, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        predictions, conv_outputs = grad_model(img_tensor, training=False)
        predicted_class = tf.argmax(predictions[0], output_type=tf.int32)
        score = tf.gather(predictions[0], predicted_class)

    grads = tape.gradient(score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return predictions[0].numpy(), heatmap.numpy()


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    from PIL import Image

    height, width = original_image.shape[:2]
    resized = Image.fromarray(np.uint8(255 * heatmap)).resize(
        (width, height), Image.BILINEAR
    )
    heat = np.asarray(resized, dtype=np.float32) / 255.0
    red = np.clip(1.5 - np.abs(4.0 * heat - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * heat - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * heat - 1.0), 0.0, 1.0)
    colormap = np.stack((red, green, blue), axis=-1) * 255.0
    overlay = original_image * (1 - alpha) + colormap * alpha
    return np.uint8(np.clip(overlay, 0, 255))


def predict_with_gradcam_to_base64(
    grad_model,
    image_array: np.ndarray,
) -> tuple[np.ndarray, str]:
    from PIL import Image

    probabilities, heatmap = predict_and_generate_gradcam(grad_model, image_array)
    overlay = overlay_heatmap(image_array, heatmap)
    img = Image.fromarray(overlay)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=False)
    return probabilities, base64.b64encode(buffer.getvalue()).decode("utf-8")
