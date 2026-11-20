"""Check metric calculations against known confusion tables and saved predictions."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from classification_metrics import CIFAR10_CLASSES, _binary_ranking_metrics, classification_metrics, save_report


def test_ranking_scores_group_ties_and_compute_average_precision():
    auc, ap = _binary_ranking_metrics(np.array([0, 0, 1, 1]), np.array([.1, .4, .35, .8]))
    assert auc == pytest.approx(.75)
    assert ap == pytest.approx(5 / 6)
    assert _binary_ranking_metrics(np.array([0, 1]), np.array([.5, .5])) == (.5, .5)
    assert _binary_ranking_metrics(np.array([0, 0]), np.array([.5, .7])) == (None, None)


def test_confidence_filter_reports_recall_lost_to_rejection():
    metrics = classification_metrics(np.array([0, 1]), np.array([[2, 0], [.2, 0]]))
    threshold = next(row for row in metrics["selective_predictions"] if row["threshold"] == .7)
    assert threshold["coverage"] == .5
    assert threshold["accuracy"] == 1
    assert threshold["macro_recall_counting_rejections_as_misses"] == .5


@pytest.mark.parametrize("bins", [0, -1, 1.5, True])
def test_invalid_calibration_bin_count(bins):
    with pytest.raises(ValueError, match="positive integer"):
        classification_metrics(np.array([0]), np.array([[1, 0]]), calibration_bins=bins)


def test_report_does_not_claim_unbalanced_data_has_equal_class_support(tmp_path):
    metrics = classification_metrics(np.array([0, 0, 0, 1]), np.array([[1, 0]] * 4))
    output = tmp_path / "metrics.json"
    save_report(metrics, output, make_plot=False)
    assert "1,000 examples" not in output.with_suffix(".md").read_text(encoding="utf-8")
    collision = tmp_path / "metrics.csv"
    collision.write_text("keep this", encoding="utf-8")
    with pytest.raises(ValueError, match=".json"):
        save_report(metrics, collision, make_plot=False)
    assert collision.read_text(encoding="utf-8") == "keep this"


def test_known_unbalanced_confusion_matrix():
    labels = np.array([0, 0, 1, 1])
    logits = np.array([[3, 1], [2, 0], [2, 1], [0, 3]])
    result = classification_metrics(labels, logits, ["a", "b"])
    assert result["confusion_matrix"] == [[2, 0], [1, 1]]
    assert result["macro_precision"] == pytest.approx(5 / 6)
    assert result["macro_recall"] == pytest.approx(.75)
    assert result["macro_f1"] == pytest.approx((.8 + 2 / 3) / 2)
    assert result["cohen_kappa"] == pytest.approx(.5)
    assert result["matthews_correlation_coefficient"] == pytest.approx(1 / np.sqrt(3))
    assert result["per_class"][1]["false_negative"] == 1
    assert result["errors"] == 1


def test_weighted_and_macro_recall_differ_on_imbalanced_data():
    labels = np.array([0, 0, 0, 1])
    result = classification_metrics(labels, np.array([[1, 0]] * 4))
    assert result["accuracy"] == .75
    assert result["weighted_recall"] == .75
    assert result["macro_recall"] == .5
    assert result["per_class"][1]["precision"] == 0
    assert result["matthews_correlation_coefficient"] == 0


def test_probability_metrics_and_top_k_on_uniform_logits():
    result = classification_metrics(np.array([0, 1, 2]), np.zeros((3, 3)), calibration_bins=np.int64(3))
    assert result["negative_log_likelihood"] == pytest.approx(np.log(3))
    assert result["brier_score"] == pytest.approx(2 / 3)
    assert result["expected_calibration_error"] == pytest.approx(0)
    assert result["top_k_accuracy"] == {"1": 1 / 3, "3": 1.0}
    assert sum(item["count"] for item in result["reliability_bins"]) == 3
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("labels,logits", [
    (np.array([], dtype=int), np.empty((0, 2))),
    (np.array([2]), np.zeros((1, 2))),
    (np.array([0.5]), np.zeros((1, 2))),
    (np.array([0]), np.array([[np.nan, 0]])),
])
def test_invalid_predictions_fail(labels, logits):
    with pytest.raises(ValueError):
        classification_metrics(labels, logits)


def test_saved_metrics_are_bound_to_actual_cpp_predictions(tmp_path):
    predictions = ROOT / "docs/test_predictions.npz"
    with np.load(predictions, allow_pickle=False) as data:
        actual = classification_metrics(data["labels"], data["cpp_logits"], CIFAR10_CLASSES)
    saved = json.loads((ROOT / "docs/classification_metrics.json").read_text(encoding="utf-8"))
    assert saved["prediction_file_sha256"] == hashlib.sha256(predictions.read_bytes()).hexdigest()
    for key in ["accuracy", "macro_recall", "macro_precision", "macro_f1", "confusion_matrix", "per_class"]:
        assert saved[key] == actual[key]
    output = tmp_path / "metrics.json"
    save_report(actual, output, predictions, make_plot=False)
    assert output.with_suffix(".csv").is_file()
    assert output.with_suffix(".md").is_file()


def test_training_evaluation_retains_recall_without_changing_accuracy_api():
    import ast
    import torch
    from torch.nn import functional as F
    tree = ast.parse((ROOT / "training/train_replacement.py").read_text(encoding="utf-8"))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                   node.name in {"evaluate", "prepare_images", "ste_quantize"}]
    scope = dict(torch=torch, F=F, np=np, classification_metrics=classification_metrics,
                 CIFAR10_CLASSES=CIFAR10_CLASSES)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "train_replacement.py", "exec"), scope)

    class ConstantClassifier(torch.nn.Module):
        def forward(self, x):
            out = torch.zeros(len(x), 10)
            out[:, 0] = 1
            return out

    images = torch.zeros(4, 32, 32, 3, dtype=torch.uint8)
    labels = torch.tensor([0, 0, 0, 1])
    args = (ConstantClassifier(), images, labels, torch.arange(4), 2,
            torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1))
    assert scope["evaluate"](*args) == .75
    metrics = scope["evaluate"](*args, return_metrics=True)
    assert metrics["accuracy"] == .75
    assert metrics["macro_recall"] == .1
    assert metrics["balanced_accuracy"] == .5
    empty_args = (*args[:3], torch.empty(0, dtype=torch.int64), *args[4:])
    with pytest.raises(ValueError, match="nonempty"):
        scope["evaluate"](*empty_args)
    with pytest.raises(ValueError, match="positive"):
        scope["evaluate"](*args[:4], 0, *args[5:])
