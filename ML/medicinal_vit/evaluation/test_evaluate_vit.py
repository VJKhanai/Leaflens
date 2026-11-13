import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from .evaluate_vit import (
    calculate_metrics,
    class_labels_from_config,
    checkpoint_revision,
)


class EvaluationHelpersTests(unittest.TestCase):
    def test_metrics_include_top5_confusion_and_per_class_counts(self):
        metrics = calculate_metrics(
            [0, 1, 1],
            [[0, 1], [0, 1], [1, 0]],
            class_count=2,
        )

        self.assertAlmostEqual(metrics["top1_accuracy"], 2 / 3)
        self.assertEqual(metrics["top5_accuracy"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0], [1, 1]])
        self.assertEqual(metrics["per_class"][0]["support"], 1)
        self.assertEqual(metrics["per_class"][1]["support"], 2)

    def test_class_labels_are_read_in_checkpoint_index_order(self):
        config = SimpleNamespace(
            num_labels=80,
            id2label={str(index): f"Plant {index}" for index in range(80)},
        )

        labels = class_labels_from_config(config)

        self.assertEqual(labels[0], "Plant 0")
        self.assertEqual(labels[-1], "Plant 79")

    @patch("ML.medicinal_vit.evaluation.evaluate_vit.try_to_load_from_cache")
    def test_checkpoint_revision_uses_cached_snapshot_when_config_omits_hash(
        self,
        cache_lookup,
    ):
        cache_lookup.return_value = str(
            Path("cache") / "snapshots" / "revision-123" / "vit-plant-push" / "config.json"
        )
        config = SimpleNamespace(_commit_hash=None)
        model = SimpleNamespace(config=SimpleNamespace(_commit_hash=None))

        self.assertEqual(checkpoint_revision(config, model, None), "revision-123")


if __name__ == "__main__":
    unittest.main()
