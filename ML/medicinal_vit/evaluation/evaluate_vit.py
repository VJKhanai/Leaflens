"""Evaluate the cached medicinal-plant ViT checkpoint on verified image splits."""

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from huggingface_hub import try_to_load_from_cache
import torch
import transformers
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification


MODEL_ID = "dhivyasri-d/vit-plant-classifier"
MODEL_SUBFOLDER = "vit-plant-push"
EXPECTED_CLASS_COUNT = 80
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
EVALUATION_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = EVALUATION_ROOT / "results"


class EvaluationError(ValueError):
    """Raised when dataset inputs are not suitable for evaluation."""


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    split: str
    label: str
    label_index: int
    pixel_sha256: str


def class_labels_from_config(config):
    id2label = config.id2label
    labels = []
    for index in range(config.num_labels):
        label = id2label.get(index, id2label.get(str(index)))
        if not isinstance(label, str) or not label.strip():
            raise EvaluationError(f"Checkpoint has no label for class index {index}.")
        labels.append(label)
    if len(labels) != EXPECTED_CLASS_COUNT or len(set(labels)) != len(labels):
        raise EvaluationError(
            f"Expected {EXPECTED_CLASS_COUNT} unique checkpoint labels; found {len(labels)}."
        )
    return labels


def checkpoint_revision(config, model, requested_revision):
    for candidate in (
        getattr(model.config, "_commit_hash", None),
        getattr(config, "_commit_hash", None),
    ):
        if candidate:
            return candidate

    cache_revision = requested_revision or "main"
    cached_config = try_to_load_from_cache(
        MODEL_ID,
        f"{MODEL_SUBFOLDER}/config.json",
        revision=cache_revision,
    )
    if isinstance(cached_config, str):
        snapshot = Path(cached_config)
        return snapshot.parent.parent.name
    return requested_revision or "unavailable"


def load_dataset_metadata(dataset_root):
    metadata_path = dataset_root / "dataset_info.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise EvaluationError(
            f"Required dataset metadata is missing: {metadata_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise EvaluationError(f"Invalid JSON in {metadata_path}: {error}") from error
    if not isinstance(metadata, dict):
        raise EvaluationError("dataset_info.json must contain a JSON object.")

    required_text = (
        "dataset_name",
        "source",
        "license",
        "labeling_provenance",
        "model_training_overlap_evidence",
    )
    for key in required_text:
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise EvaluationError(f"dataset_info.json needs a non-empty '{key}' field.")
    if metadata.get("ground_truth_verified") is not True:
        raise EvaluationError(
            "dataset_info.json must attest that ground-truth labels were verified."
        )
    if metadata.get("independent_test_set") is not True:
        raise EvaluationError(
            "dataset_info.json must attest that the test split is independent."
        )
    if metadata.get("model_training_overlap_reviewed") is not True:
        raise EvaluationError(
            "Review possible overlap with model training data and set "
            "'model_training_overlap_reviewed' to true."
        )
    return metadata


def _pixel_sha256(path):
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
            digest = hashlib.sha256()
            digest.update(f"{image.width}x{image.height}:".encode("ascii"))
            digest.update(image.tobytes())
            return digest.hexdigest()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise EvaluationError(f"Cannot read image '{path}': {error}") from error


def inspect_dataset(dataset_root, labels, metadata):
    class_to_index = {label: index for index, label in enumerate(labels)}
    raw_mapping = metadata.get("class_name_mapping", {})
    if not isinstance(raw_mapping, dict) or any(
        not isinstance(source, str)
        or not isinstance(target, str)
        or target not in class_to_index
        for source, target in raw_mapping.items()
    ):
        raise EvaluationError(
            "class_name_mapping must map source directory names to checkpoint labels."
        )

    records = []
    inventory = []
    for split in SPLITS:
        split_root = dataset_root / split
        if not split_root.is_dir():
            raise EvaluationError(
                f"Dataset must contain '{split}' split directory: {split_root}"
            )

        class_dirs = sorted(path for path in split_root.iterdir() if path.is_dir())
        if not class_dirs:
            raise EvaluationError(f"No class directories found under '{split_root}'.")

        split_folders = {}
        for class_dir in class_dirs:
            label = raw_mapping.get(class_dir.name, class_dir.name)
            if label not in class_to_index:
                raise EvaluationError(
                    f"Unknown class directory '{class_dir.name}' in {split}; "
                    "use a verified class_name_mapping if its label is equivalent."
                )
            previous_name = split_folders.get(label)
            if previous_name is not None:
                raise EvaluationError(
                    f"Both '{previous_name}' and '{class_dir.name}' map to '{label}' "
                    f"in the {split} split."
                )
            split_folders[label] = class_dir.name

        missing_labels = sorted(set(labels) - set(split_folders))
        if missing_labels:
            raise EvaluationError(
                f"The {split} split is missing checkpoint classes: "
                + ", ".join(missing_labels)
            )

        for label in labels:
            folder_name = split_folders[label]
            image_paths = sorted(
                path
                for path in (split_root / folder_name).rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            inventory.append({
                "split": split,
                "class_name": label,
                "image_count": len(image_paths),
            })
            for path in image_paths:
                records.append(ImageRecord(
                    path=path,
                    split=split,
                    label=label,
                    label_index=class_to_index[label],
                    pixel_sha256=_pixel_sha256(path),
                ))

    test_counts = {
        row["class_name"]: row["image_count"]
        for row in inventory
        if row["split"] == "test"
    }
    empty_test_classes = [label for label in labels if test_counts.get(label, 0) == 0]
    if empty_test_classes:
        raise EvaluationError(
            "The test split needs at least one image for every checkpoint class; "
            "empty classes: " + ", ".join(empty_test_classes)
        )

    duplicates_by_hash = defaultdict(list)
    for record in records:
        duplicates_by_hash[record.pixel_sha256].append(record)
    duplicates = []
    for digest, matches in sorted(duplicates_by_hash.items()):
        if len(matches) > 1:
            for match in matches:
                duplicates.append({
                    "pixel_sha256": digest,
                    "split": match.split,
                    "class_name": match.label,
                    "path": str(match.path.relative_to(dataset_root)),
                })

    return records, inventory, duplicates


def calculate_metrics(true_indices, top_indices, class_count):
    if not true_indices or len(true_indices) != len(top_indices):
        raise EvaluationError("No complete prediction records are available.")

    confusion = [[0 for _ in range(class_count)] for _ in range(class_count)]
    top1_correct = 0
    top5_correct = 0
    for true_index, ranked_indices in zip(true_indices, top_indices):
        predicted_index = ranked_indices[0]
        confusion[true_index][predicted_index] += 1
        top1_correct += predicted_index == true_index
        top5_correct += true_index in ranked_indices[:5]

    per_class = []
    for index in range(class_count):
        true_positive = confusion[index][index]
        support = sum(confusion[index])
        predicted_count = sum(row[index] for row in confusion)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class.append({
            "class_index": index,
            "support": support,
            "predicted_count": predicted_count,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        })

    return {
        "top1_accuracy": top1_correct / len(true_indices),
        "top5_accuracy": top5_correct / len(true_indices),
        "macro_precision": sum(row["precision"] for row in per_class) / class_count,
        "macro_recall": sum(row["recall"] for row in per_class) / class_count,
        "macro_f1_score": sum(row["f1_score"] for row in per_class) / class_count,
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _create_run_directory(results_root):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _save_inventory(run_dir, inventory, dataset_root, duplicates):
    _write_csv(
        run_dir / "dataset_inventory.csv",
        ("split", "class_name", "image_count"),
        inventory,
    )
    _write_csv(
        run_dir / "duplicate_images.csv",
        ("pixel_sha256", "split", "class_name", "path"),
        duplicates,
    )
    if duplicates:
        (run_dir / "readiness.json").write_text(
            json.dumps({
                "status": "blocked",
                "reason": "Identical decoded RGB images occur more than once across dataset splits.",
                "dataset_path": str(dataset_root),
                "duplicate_records": len(duplicates),
            }, indent=2) + "\n",
            encoding="utf-8",
        )


def _evaluate_model(records, processor, model, labels, batch_size):
    label_to_index = {label: index for index, label in enumerate(labels)}
    test_records = [record for record in records if record.split == "test"]
    true_indices = []
    top_indices = []
    incorrect = []

    model.eval()
    for start in range(0, len(test_records), batch_size):
        batch_records = test_records[start : start + batch_size]
        images = []
        for record in batch_records:
            try:
                with Image.open(record.path) as source:
                    images.append(source.convert("RGB"))
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
                raise EvaluationError(f"Cannot reload image '{record.path}': {error}") from error

        inputs = processor(images=images, return_tensors="pt")
        inputs = {name: tensor.to(torch.device("cpu")) for name, tensor in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
            top_values, top_class_indices = torch.topk(probabilities, k=5, dim=-1)

        for offset, record in enumerate(batch_records):
            ranked_indices = top_class_indices[offset].tolist()
            true_index = label_to_index[record.label]
            true_indices.append(true_index)
            top_indices.append(ranked_indices)
            if ranked_indices[0] != true_index:
                incorrect.append({
                    "path": str(record.path),
                    "ground_truth": record.label,
                    "predicted_top1": labels[ranked_indices[0]],
                    "top1_confidence_percent": round(
                        float(top_values[offset][0].item()) * 100, 4
                    ),
                    "top5": json.dumps([
                        {
                            "class_name": labels[class_index],
                            "confidence_percent": round(
                                float(top_values[offset][rank].item()) * 100, 4
                            ),
                        }
                        for rank, class_index in enumerate(ranked_indices)
                    ], ensure_ascii=False),
                })

    return true_indices, top_indices, incorrect


def _save_metrics(run_dir, metrics, labels, incorrect):
    per_class_rows = [
        {"class_name": labels[row["class_index"]], **{
            key: row[key]
            for key in ("support", "predicted_count", "precision", "recall", "f1_score")
        }}
        for row in metrics["per_class"]
    ]
    _write_csv(
        run_dir / "per_class_metrics.csv",
        ("class_name", "support", "predicted_count", "precision", "recall", "f1_score"),
        per_class_rows,
    )
    confusion_rows = []
    for index, row in enumerate(metrics["confusion_matrix"]):
        confusion_rows.append({
            "ground_truth_label": labels[index],
            **{labels[column]: row[column] for column in range(len(labels))},
        })
    _write_csv(
        run_dir / "confusion_matrix.csv",
        ("ground_truth_label", *labels),
        confusion_rows,
    )
    _write_csv(
        run_dir / "incorrect_predictions.csv",
        ("path", "ground_truth", "predicted_top1", "top1_confidence_percent", "top5"),
        incorrect,
    )


def run_evaluation(dataset_path, results_path, batch_size, revision=None):
    dataset_root = Path(dataset_path).expanduser().resolve()
    if not dataset_root.is_dir():
        raise EvaluationError(f"Dataset directory does not exist: {dataset_root}")
    if batch_size < 1:
        raise EvaluationError("--batch-size must be at least 1.")

    metadata = load_dataset_metadata(dataset_root)
    load_kwargs = {"subfolder": MODEL_SUBFOLDER, "local_files_only": True}
    if revision:
        load_kwargs["revision"] = revision
    config = AutoConfig.from_pretrained(MODEL_ID, **load_kwargs)
    labels = class_labels_from_config(config)
    records, inventory, duplicates = inspect_dataset(dataset_root, labels, metadata)

    results_root = Path(results_path).expanduser().resolve()
    run_dir = _create_run_directory(results_root)
    _save_inventory(run_dir, inventory, dataset_root, duplicates)
    if duplicates:
        raise EvaluationError(
            f"Found {len(duplicates)} duplicate-image records across dataset splits. "
            f"Review {run_dir / 'duplicate_images.csv'} before evaluation."
        )

    processor = AutoImageProcessor.from_pretrained(MODEL_ID, **load_kwargs)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID, **load_kwargs)
    model.to(torch.device("cpu"))
    if model.config.num_labels != EXPECTED_CLASS_COUNT:
        raise EvaluationError(
            f"Checkpoint has {model.config.num_labels} classes; "
            f"expected {EXPECTED_CLASS_COUNT}."
        )
    config_revision = getattr(config, "_commit_hash", None)
    model_revision = getattr(model.config, "_commit_hash", None)
    if config_revision and model_revision and config_revision != model_revision:
        raise EvaluationError("Loaded checkpoint config and weights have different revisions.")
    checkpoint_revision_value = checkpoint_revision(config, model, revision)

    true_indices, top_indices, incorrect = _evaluate_model(
        records, processor, model, labels, batch_size
    )
    metrics = calculate_metrics(true_indices, top_indices, len(labels))
    _save_metrics(run_dir, metrics, labels, incorrect)

    test_count = sum(record.split == "test" for record in records)
    summary = {
        "status": "completed",
        "evaluation_date_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_root),
        "dataset": metadata,
        "test_split": "test",
        "tested_images": test_count,
        "classes": len(labels),
        "model_id": MODEL_ID,
        "model_subfolder": MODEL_SUBFOLDER,
        "checkpoint_revision": checkpoint_revision_value,
        "processor": processor.__class__.__name__,
        "processor_size": getattr(processor, "size", None),
        "device": "cpu",
        "batch_size": batch_size,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "top1_accuracy": metrics["top1_accuracy"],
        "top5_accuracy": metrics["top5_accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1_score": metrics["macro_f1_score"],
        "incorrect_predictions": len(incorrect),
        "duplicate_scan": {
            "method": "SHA-256 of decoded RGB pixels and image dimensions",
            "duplicate_records": 0,
            "near_duplicate_detection": False,
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return run_dir, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Root containing dataset_info.json and train/val/test class folders.",
    )
    parser.add_argument(
        "--results",
        default=str(DEFAULT_RESULTS_DIR),
        help="Output root; each run is saved in a new timestamped directory.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--revision",
        help="Optional Hugging Face commit/tag. Defaults to the cached model revision.",
    )
    args = parser.parse_args(argv)

    try:
        run_dir, summary = run_evaluation(
            args.dataset,
            args.results,
            args.batch_size,
            revision=args.revision,
        )
    except EvaluationError as error:
        print(f"Evaluation blocked: {error}", file=sys.stderr)
        return 2

    print(f"Results: {run_dir}")
    print(f"Test images: {summary['tested_images']}")
    print(f"Top-1 accuracy: {summary['top1_accuracy'] * 100:.2f}%")
    print(f"Top-5 accuracy: {summary['top5_accuracy'] * 100:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
