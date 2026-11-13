# Medicinal ViT evaluation

This evaluation code is separate from the production prediction service. It
loads `dhivyasri-d/vit-plant-classifier` from its `vit-plant-push` subfolder
with the checkpoint's `AutoImageProcessor`, runs CPU inference, and writes each
run to a new timestamped directory under `results/`. It does not fine-tune or
save model weights.

## Dataset requirements

Evaluation is blocked until an independently sourced, ground-truth-verified
dataset is provided. Supply a directory with this layout:

```text
dataset/
  dataset_info.json
  train/
    <exact checkpoint label>/
      image.jpg
  val/
    <exact checkpoint label>/
      image.jpg
  test/
    <exact checkpoint label>/
      image.jpg
```

Each split must contain folders for all 80 checkpoint labels, with at least one
test image in every class. Folder labels must match the checkpoint's `id2label`
names. A documented, curator-verified `class_name_mapping` may map source
folder names to checkpoint labels; labels are never guessed or normalized by
the script.

`dataset_info.json` must identify the dataset and document the label source,
independence and training-overlap review. Example:

```json
{
  "dataset_name": "Name of the independently collected dataset",
  "source": "Dataset URL or collection record",
  "citation": "Optional citation",
  "license": "Dataset license or permission",
  "labeling_provenance": "How each image's plant label was verified",
  "ground_truth_verified": true,
  "independent_test_set": true,
  "model_training_overlap_reviewed": true,
  "model_training_overlap_evidence": "Evidence that test images were not used to train this checkpoint",
  "class_name_mapping": {}
}
```

These metadata assertions are not independently authenticated by the script;
the dataset curator must verify them. All three splits are required so the
script can check decoded-pixel duplicates across train, validation and test.
Runs with exact duplicate images are blocked and list them in
`duplicate_images.csv`. The duplicate check detects identical decoded pixels,
not visually similar near-duplicates; review source provenance and any
perceptual-duplicate reports separately.

## Run

Activate the existing evaluation environment from the repository root:

```powershell
.\vit-venv\Scripts\Activate.ps1
python -m ML.medicinal_vit.evaluation.evaluate_vit --dataset D:\path\to\verified-dataset
```

The script uses cached model files only by default. An optional `--revision`
pins a Hugging Face commit or tag. Each successful run records the resolved
checkpoint revision, dataset path and metadata, UTC date, processor details,
runtime versions, top-1/top-5 accuracy, macro metrics, and tested-image count.
It writes `summary.json`, `dataset_inventory.csv`, `per_class_metrics.csv`,
`confusion_matrix.csv`, `incorrect_predictions.csv`, and
`duplicate_images.csv` under a unique `results/<run-id>/` directory.

## Repository dataset audit (2026-11-13)

No medicinal ViT evaluation image dataset is present in this checkout. `ML/data`
and the manifest-recorded `B:\LeafLens-AI\ML\data` paths are absent. The
cached ViT snapshot currently resolves to revision
`2827af6b48b0562778cf0a45191a67b816013492`; its config has 80 labels and its
image processor resizes to 224x224. The
repository has split manifests for a separate 78-class dataset
(`grouped_manifest.csv`: 4,885 train, 1,023 validation, 935 test rows), but all
recorded image paths are unavailable here. The manifest's per-class counts
range from 28 to 126 (train), 2 to 27 (validation), and 2 to 25 (test); the
medians are 56.5, 12, and 11. Its labels do not fully match the 80-class ViT
checkpoint: Lemongrass and Pepper are absent, and `Astma_weed`, `Balloon_Vine`,
`Malabar_Nut`, and `Malabar_Spinach` differ from the checkpoint's space-separated
labels. Those names need verified mapping, and the two absent classes need
actual labeled images. The manifests are not sufficient ground truth for
evaluating this checkpoint.

The grouped manifest's `group_id` values do not cross splits, but a separate
duplicate report records 131 potential near-duplicate groups across splits
(294 image records, spanning 44 labels). Because their referenced images are
unavailable, these overlaps cannot be verified in this checkout. The ten image
files under `ML/` are duplicate-review materials or a misclassification
example, not a labeled 80-class evaluation set. No metrics or evaluation result
files are produced until a suitable dataset is supplied.
