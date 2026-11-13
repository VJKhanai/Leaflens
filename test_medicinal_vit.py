
import sys
import torch

from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)

MODEL_ID = "dhivyasri-d/vit-plant-classifier"

print("=" * 55)
print("LEAFLENS AI - MEDICINAL PLANT VIT")
print("=" * 55)

processor = AutoImageProcessor.from_pretrained(
    MODEL_ID,
    subfolder="vit-plant-push"
)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_ID,
    subfolder="vit-plant-push"
)

model.eval()

print("\nMODEL INFORMATION")
print("Architecture:", model.config.model_type)
print("Number of classes:", model.config.num_labels)
print("Image configuration:", processor.size)
print("Device: CPU")

# Display classes
print("\nAVAILABLE PLANT CLASSES")

for index in range(model.config.num_labels):
    label = model.config.id2label[index]
    print(f"{index}: {label}")

# Optional image prediction
if len(sys.argv) > 1:
    image_path = sys.argv[1]

    print("\nLoading image:", image_path)

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.inference_mode():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    top_predictions = torch.topk(
        probabilities,
        k=min(5, model.config.num_labels)
    )

    print("\nTOP PREDICTIONS")

    for score, index in zip(
        top_predictions.values,
        top_predictions.indices
    ):
        label = model.config.id2label[index.item()]

        print(
            f"{label}: {score.item() * 100:.2f}%"
        )

print("\nMODEL TEST COMPLETED")
