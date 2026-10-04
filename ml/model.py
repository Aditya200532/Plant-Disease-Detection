from __future__ import annotations


def build_model(
    num_classes: int,
    image_size: int = 224,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
    weights: str | None = "imagenet",
):
    import tensorflow as tf

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="training_augmentation",
    )
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights=weights,
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = tf.keras.Model(inputs, outputs, name="plant_disease_mobilenet_v2")
    compile_model(model, learning_rate, num_classes)
    return model, base_model


def compile_model(model, learning_rate: float, num_classes: int) -> None:
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=min(3, num_classes), name="top_3_accuracy"
            ),
        ],
    )


def enable_fine_tuning(base_model, trainable_layers: int) -> None:
    import tensorflow as tf

    base_model.trainable = True
    cutoff = max(0, len(base_model.layers) - trainable_layers)
    for index, layer in enumerate(base_model.layers):
        layer.trainable = index >= cutoff and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
