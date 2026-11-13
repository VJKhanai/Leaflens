"""CPU inference service for the cached medicinal-plant ViT checkpoint."""

from PIL import Image, UnidentifiedImageError
import torch
from huggingface_hub.errors import HfHubHTTPError
from transformers import AutoImageProcessor, AutoModelForImageClassification


MODEL_ID = "dhivyasri-d/vit-plant-classifier"
MODEL_SUBFOLDER = "vit-plant-push"
EXPECTED_CLASS_COUNT = 80
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "BMP"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/x-ms-bmp"}


class ImageInputError(ValueError):
    """Raised when an uploaded image is missing, unsupported, or invalid."""


class ModelInferenceError(RuntimeError):
    """Raised when a valid image cannot be classified by the loaded model."""


class ModelLoadError(RuntimeError):
    """Raised when the Hugging Face model or processor cannot be loaded."""


class MedicinalViTService:
    def __init__(self):
        self.device = torch.device("cpu")
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                MODEL_ID,
                subfolder=MODEL_SUBFOLDER,
            )
            self.model = AutoModelForImageClassification.from_pretrained(
                MODEL_ID,
                subfolder=MODEL_SUBFOLDER,
            ).to(self.device)
        except (HfHubHTTPError, OSError):
            raise ModelLoadError(
                f"Unable to load {MODEL_ID} (subfolder {MODEL_SUBFOLDER}). "
                "Check network access or the Hugging Face cache. If model access "
                "is restricted, authenticate with `hf auth login`."
            ) from None
        self.model.eval()

        config = self.model.config
        if config.model_type != "vit":
            raise RuntimeError(f"Expected a ViT checkpoint, got {config.model_type!r}.")
        if config.num_labels != EXPECTED_CLASS_COUNT:
            raise RuntimeError(
                f"Expected {EXPECTED_CLASS_COUNT} model classes, got {config.num_labels}."
            )

        labels = config.id2label
        self.class_metadata = []
        for index in range(config.num_labels):
            label = labels.get(index, labels.get(str(index)))
            if not isinstance(label, str) or not label.strip():
                raise RuntimeError(f"The model has no class name for output index {index}.")
            self.class_metadata.append({"index": index, "class_name": label})

    @property
    def model_loaded(self):
        return self.model is not None and self.processor is not None

    def health(self):
        return {
            "status": "healthy" if self.model_loaded else "unavailable",
            "model_loaded": self.model_loaded,
            "model_id": MODEL_ID,
            "architecture": "Vision Transformer (ViT)",
            "model_type": self.model.config.model_type,
            "class_count": len(self.class_metadata),
            "device": str(self.device),
        }

    def predict_upload(self, uploaded_file, include_all=False):
        if uploaded_file is None:
            raise ImageInputError("Upload an image using the 'file' field.")
        if not uploaded_file.filename:
            raise ImageInputError("Choose an image file to upload.")
        if uploaded_file.mimetype.lower() not in ALLOWED_MIME_TYPES:
            raise ImageInputError("Unsupported image type. Upload a JPG, PNG, or BMP image.")

        try:
            with Image.open(uploaded_file.stream) as source:
                if source.format not in ALLOWED_IMAGE_FORMATS:
                    raise ImageInputError(
                        "Unsupported image format. Upload a JPG, PNG, or BMP image."
                    )
                image = source.convert("RGB")
        except ImageInputError:
            raise
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
            raise ImageInputError("The uploaded file is not a valid readable image.") from error

        return self.predict_image(image, include_all=include_all)

    def predict_image(self, image, include_all=False):
        try:
            inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
            inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
            with torch.inference_mode():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits[0], dim=-1)
            if probabilities.numel() != len(self.class_metadata):
                raise ModelInferenceError(
                    "The model output class count does not match its class metadata."
                )
        except ModelInferenceError:
            raise
        except (RuntimeError, TypeError, ValueError, KeyError) as error:
            raise ModelInferenceError("The model could not classify this image.") from error

        ranked_indices = torch.argsort(probabilities, descending=True).tolist()
        predictions = [
            {
                "class_index": index,
                "class_name": self.class_metadata[index]["class_name"],
                "confidence_percent": round(float(probabilities[index].item()) * 100, 2),
            }
            for index in ranked_indices
        ]
        result = {
            "predictions": predictions[:5],
            "class_count": len(self.class_metadata),
        }
        if include_all:
            result["all_predictions"] = predictions
        return result
