from __future__ import annotations

import json
import numpy as np
from pathlib import Path


class Predictor:
    UNKNOWN_CONFIDENCE = 0.6
    UNKNOWN_MARGIN = 0.2

    def __init__(self, model_path: str, metadata: dict):
        import tensorflow as tf
        self.model = tf.keras.models.load_model(model_path)
        self.class_names: list[str] = metadata["class_names"]
        self.image_size: int = metadata["image_size"]
        self.num_classes: int = metadata["number_of_classes"]

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        import tensorflow as tf
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, (self.image_size, self.image_size), antialias=True)
        return image.numpy()

    def predict(self, image_bytes: bytes) -> dict:
        image_array = self.preprocess(image_bytes)
        input_batch = np.expand_dims(image_array, axis=0).astype(np.float32)
        probabilities = self.model.predict(input_batch, verbose=0)[0]
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

        # Softmax always picks one of the trained classes, so weak or ambiguous scores mean "not a known leaf".
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

        return result, image_array

    def supported_plants(self) -> list[str]:
        return sorted({self._parse_class_name(name)[0] for name in self.class_names})

    def predict_with_gradcam(self, image_bytes: bytes) -> dict:
        result, image_array = self.predict(image_bytes)
        try:
            from ml.gradcam import gradcam_to_base64
            gradcam_image = gradcam_to_base64(
                self.model, image_array, result["predicted_index"]
            )
            result["gradcam"] = gradcam_image
        except Exception:
            pass
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
