import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from flask import Flask
from PIL import Image
import torch

from .routes import create_medicinal_blueprint
from .service import (
    EXPECTED_CLASS_COUNT,
    MODEL_ID,
    MODEL_SUBFOLDER,
    ImageInputError,
    ModelLoadError,
    MedicinalViTService,
)


class FakeProcessor:
    def __call__(self, images, return_tensors):
        if images.mode != "RGB":
            raise AssertionError("Expected RGB input")
        return {"pixel_values": torch.zeros((1, 3, 224, 224))}


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(
            model_type="vit",
            num_labels=EXPECTED_CLASS_COUNT,
            id2label={index: f"Plant {index}" for index in range(EXPECTED_CLASS_COUNT)},
        )
        self.inference_mode_was_enabled = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True
        return self

    def __call__(self, **_inputs):
        self.inference_mode_was_enabled = torch.is_inference_mode_enabled()
        logits = torch.arange(EXPECTED_CLASS_COUNT, dtype=torch.float).unsqueeze(0)
        return SimpleNamespace(logits=logits)


def make_service():
    service = object.__new__(MedicinalViTService)
    service.device = torch.device("cpu")
    service.processor = FakeProcessor()
    service.model = FakeModel()
    service.class_metadata = [
        {"index": index, "class_name": f"Plant {index}"}
        for index in range(EXPECTED_CLASS_COUNT)
    ]
    return service


def make_png():
    buffer = io.BytesIO()
    Image.new("RGBA", (8, 8), color=(1, 2, 3, 255)).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


class MedicinalViTServiceTests(unittest.TestCase):
    @patch("backend.medicinal_vit.service.AutoImageProcessor.from_pretrained")
    @patch("backend.medicinal_vit.service.AutoModelForImageClassification.from_pretrained")
    def test_loads_processor_and_model_once_from_requested_subfolder(
        self,
        load_model,
        load_processor,
    ):
        fake_model = FakeModel()
        load_model.return_value = fake_model
        load_processor.return_value = FakeProcessor()

        service = MedicinalViTService()

        load_processor.assert_called_once_with(MODEL_ID, subfolder=MODEL_SUBFOLDER)
        load_model.assert_called_once_with(MODEL_ID, subfolder=MODEL_SUBFOLDER)
        self.assertTrue(fake_model.evaluated)
        self.assertEqual(service.health()["class_count"], EXPECTED_CLASS_COUNT)

    @patch(
        "backend.medicinal_vit.service.AutoImageProcessor.from_pretrained",
        side_effect=OSError("checkpoint unavailable"),
    )
    def test_reports_how_to_recover_when_checkpoint_is_unavailable(self, _load_processor):
        with self.assertRaises(ModelLoadError) as error:
            MedicinalViTService()

        self.assertIn(MODEL_ID, str(error.exception))
        self.assertIn("Hugging Face cache", str(error.exception))
        self.assertIn("hf auth login", str(error.exception))

    def test_returns_top_five_and_optional_full_predictions_in_inference_mode(self):
        service = make_service()

        result = service.predict_image(
            Image.new("RGBA", (8, 8)),
            include_all=True,
        )

        self.assertEqual(len(result["predictions"]), 5)
        self.assertEqual(result["predictions"][0]["class_name"], "Plant 79")
        self.assertEqual(result["predictions"][0]["class_index"], 79)
        self.assertGreaterEqual(result["predictions"][0]["confidence_percent"], 0)
        self.assertEqual(len(result["all_predictions"]), EXPECTED_CLASS_COUNT)
        self.assertEqual(result["all_predictions"][-1]["class_name"], "Plant 0")
        self.assertEqual(result["class_count"], EXPECTED_CLASS_COUNT)
        self.assertTrue(service.model.inference_mode_was_enabled)

    def test_rejects_invalid_uploaded_images(self):
        service = make_service()

        with self.assertRaises(ImageInputError):
            service.predict_upload(
                SimpleNamespace(filename="plant.gif", mimetype="image/gif", stream=io.BytesIO())
            )
        with self.assertRaises(ImageInputError):
            service.predict_upload(
                SimpleNamespace(filename="plant.png", mimetype="image/png", stream=io.BytesIO(b"bad"))
            )

    def test_independent_health_and_prediction_routes(self):
        service = make_service()
        app = Flask(__name__)
        app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
        app.register_blueprint(create_medicinal_blueprint(service))
        client = app.test_client()

        health = client.get("/medicinal/health")
        prediction = client.post(
            "/medicinal/predict?include_all=true",
            data={"file": (make_png(), "plant.png")},
            content_type="multipart/form-data",
        )

        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.get_json()["model_loaded"])
        self.assertEqual(health.get_json()["architecture"], "Vision Transformer (ViT)")
        self.assertEqual(prediction.status_code, 200)
        self.assertEqual(len(prediction.get_json()["predictions"]), 5)
        self.assertEqual(len(prediction.get_json()["all_predictions"]), EXPECTED_CLASS_COUNT)


if __name__ == "__main__":
    unittest.main()
