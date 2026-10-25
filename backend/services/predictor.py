from __future__ import annotations

import logging
import os

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")

LOGGER = logging.getLogger(__name__)


class Predictor:
    UNKNOWN_CONFIDENCE = 0.6
    UNKNOWN_MARGIN = 0.2

    def __init__(self, model_path: str, metadata: dict):
        import tensorflow as tf

        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.class_names: list[str] = metadata["class_names"]
        self.image_size: int = metadata["image_size"]
        self.num_classes: int = metadata["number_of_classes"]
        from ml.gradcam import build_grad_model
        self._grad_model = build_grad_model(self.model)

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        import tensorflow as tf
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, (self.image_size, self.image_size), antialias=True)
        return image.numpy()

    def predict(self, image_bytes: bytes) -> tuple[dict, np.ndarray]:
        image_array = self.preprocess(image_bytes)
        input_batch = image_array[np.newaxis, ...].astype(np.float32)
        probabilities = self.model(input_batch, training=False).numpy()[0]
        return self._build_result(probabilities), image_array

    def _build_result(self, probabilities: np.ndarray) -> dict:
        ranked = np.argsort(probabilities)[::-1]
        predicted_index = int(ranked[0])
        confidence = float(probabilities[predicted_index])
        margin = confidence - float(probabilities[ranked[1]])
        class_name = self.class_names[predicted_index]
        plant, disease, status = self._parse_class_name(class_name)

        top_predictions = []
        for index in ranked[:3]:
            top_plant, top_disease, _ = self._parse_class_name(self.class_names[int(index)])
            top_predictions.append({
                "plant": top_plant,
                "disease": top_disease,
                "confidence": round(float(probabilities[int(index)]), 4),
            })

        result = {
            "plant": plant,
            "disease": disease,
            "status": status,
            "confidence": round(confidence, 4),
            "class_name": class_name,
            "predicted_index": predicted_index,
            "top_predictions": top_predictions,
            "supported_plants": self.supported_plants(),
        }

        if confidence < self.UNKNOWN_CONFIDENCE or margin < self.UNKNOWN_MARGIN:
            result.update({
                "plant": "Unrecognized",
                "disease": "Not in trained classes",
                "status": "Unrecognized",
                "warning": (
                    "This leaf does not closely match any plant or disease the model was trained on. "
                    "The model supports only: " + ", ".join(self.supported_plants()) + ". "
                    "If this is one of those plants, try a clearer, well-lit photo of a single leaf."
                ),
            })
        elif confidence < 0.8:
            result["warning"] = (
                "Prediction confidence is moderate. "
                "Please verify with a clearer image of the leaf."
            )

        return result

    def supported_plants(self) -> list[str]:
        return sorted({self._parse_class_name(name)[0] for name in self.class_names})

    def predict_with_gradcam(self, image_bytes: bytes) -> dict:
        image_array = self.preprocess(image_bytes)
        try:
            from ml.gradcam import predict_with_gradcam_to_base64
            probabilities, gradcam_image = predict_with_gradcam_to_base64(
                self._grad_model, image_array
            )
            result = self._build_result(probabilities)
            result["gradcam"] = gradcam_image
        except Exception as exc:
            LOGGER.exception("Grad-CAM generation failed")
            input_batch = image_array[np.newaxis, ...].astype(np.float32)
            probabilities = self.model(input_batch, training=False).numpy()[0]
            result = self._build_result(probabilities)
            result["gradcam_error"] = f"{type(exc).__name__}: {exc}"
        return result

    @staticmethod
    def _parse_class_name(class_name: str) -> tuple[str, str, str]:
        separators = ["___", "__"]
        plant = class_name
        disease = class_name
        for sep in separators:
            if sep in class_name:
                parts = class_name.split(sep, 1)
                plant = " ".join(parts[0].replace("_", " ").split())
                disease = parts[1].replace("_", " ").strip()
                break
        else:
            if "_" in class_name:
                tokens = class_name.split("_")
                plant = tokens[0]
                disease = " ".join(tokens[1:])

        if "healthy" in disease.lower():
            status = "Healthy"
            disease = "Healthy"
        else:
            status = "Diseased"

        return plant, disease, status
