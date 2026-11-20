"""Classification metrics derived from labels and logits; no training required."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def _ratio(numerator, denominator):
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=np.float64),
                     where=np.asarray(denominator) != 0)


def _binary_ranking_metrics(target, scores):
    """One-vs-rest ROC-AUC and non-interpolated AP, grouping score ties."""
    order = np.argsort(-scores, kind="stable")
    truth = np.asarray(target, dtype=np.int64)[order]
    sorted_scores = np.asarray(scores)[order]
    positives = int(truth.sum())
    negatives = len(truth) - positives
    if not positives:
        return None, None
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(truth) - 1]
    tp = np.cumsum(truth)[ends].astype(float)
    fp = (ends + 1) - tp
    recall = np.r_[0., tp / positives]
    ap = float(np.sum(np.diff(recall) * tp / (tp + fp)))
    if not negatives:
        return None, ap
    fpr = np.r_[0., fp / negatives]
    auc = float(np.sum(np.diff(fpr) * (recall[1:] + recall[:-1]) / 2))
    return auc, ap


def classification_metrics(labels, logits, class_names=None, calibration_bins=15):
    labels = np.asarray(labels)
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("Expected nonempty [samples, classes] logits")
    n, classes = logits.shape
    if labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Expected one integer label per sample")
    if not np.isfinite(logits).all() or np.any((labels < 0) | (labels >= classes)):
        raise ValueError("Invalid labels or nonfinite logits")
    labels = labels.astype(np.int64)
    if isinstance(calibration_bins, (bool, np.bool_)) or not isinstance(calibration_bins, (int, np.integer)) or calibration_bins < 1:
        raise ValueError("calibration_bins must be a positive integer")
    calibration_bins = int(calibration_bins)
    names = list(class_names) if class_names is not None else [str(i) for i in range(classes)]
    if len(names) != classes or not all(isinstance(name, str) for name in names) or len(set(names)) != classes:
        raise ValueError("Expected a distinct string name for each class")
    predicted = logits.argmax(1)
    matrix = np.bincount(labels * classes + predicted, minlength=classes * classes).reshape(classes, classes)
    support = matrix.sum(1)
    predicted_count = matrix.sum(0)
    tp = matrix.diagonal()
    fp, fn = predicted_count - tp, support - tp
    tn = n - tp - fp - fn
    precision = _ratio(tp, predicted_count)
    recall = _ratio(tp, support)
    f1 = _ratio(2 * precision * recall, precision + recall)
    specificity = _ratio(tn, tn + fp)
    correct = int(tp.sum())
    accuracy = correct / n
    expected_agreement = float(np.dot(support.astype(float), predicted_count)) / n ** 2
    denominator = np.sqrt((float(n) ** 2 - np.dot(support.astype(float), support)) *
                          (float(n) ** 2 - np.dot(predicted_count.astype(float), predicted_count)))
    mcc = (correct * n - float(np.dot(support.astype(float), predicted_count))) / denominator if denominator else 0.0
    kappa = (accuracy - expected_agreement) / (1 - expected_agreement) if expected_agreement < 1 else 0.0
    present = support > 0
    z = 1.959963984540054
    divisor = 1 + z * z / n
    center = (accuracy + z * z / (2 * n)) / divisor
    radius = z * np.sqrt(accuracy * (1 - accuracy) / n + z * z / (4 * n * n)) / divisor
    shifted = logits - logits.max(1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
    probs = np.exp(log_probs)
    confidence = probs.max(1)
    correct_mask = predicted == labels
    brier = np.mean(np.sum(probs ** 2, axis=1) - 2 * probs[np.arange(n), labels] + 1)
    bin_ids = np.minimum((confidence * calibration_bins).astype(int), calibration_bins - 1)
    reliability, ece = [], 0.0
    for index in range(calibration_bins):
        mask = bin_ids == index
        count = int(mask.sum())
        avg_conf = float(confidence[mask].mean()) if count else None
        avg_acc = float(correct_mask[mask].mean()) if count else None
        if count:
            ece += count / n * abs(avg_conf - avg_acc)
        reliability.append(dict(lower=index / calibration_bins, upper=(index + 1) / calibration_bins,
                                count=count, confidence=avg_conf, accuracy=avg_acc))
    ranked = np.argsort(-logits, axis=1, kind="stable")
    true_rank = np.argmax(ranked == labels[:, None], axis=1) + 1
    ranking = [_binary_ranking_metrics(labels == i, probs[:, i]) for i in range(classes)]
    valid_auc = [auc for auc, _ in ranking if auc is not None]
    valid_ap = [ap for _, ap in ranking if ap is not None]
    selective = []
    for threshold in [0., .5, .7, .8, .9, .95, .99]:
        accepted = confidence >= threshold
        accepted_count = int(accepted.sum())
        retained_tp = np.bincount(labels[accepted & correct_mask], minlength=classes)
        selective.append(dict(threshold=threshold, accepted=accepted_count,
                              rejected=n - accepted_count, coverage=accepted_count / n,
                              accuracy=float(correct_mask[accepted].mean()) if accepted_count else None,
                              macro_recall_counting_rejections_as_misses=float(_ratio(retained_tp, support).mean())))
    top_k = {str(k): float(np.mean(np.any(ranked[:, :k] == labels[:, None], axis=1)))
             for k in sorted({1, min(3, classes), min(5, classes)})}
    per_class = [dict(index=i, name=names[i], support=int(support[i]),
                      predicted_count=int(predicted_count[i]), true_positive=int(tp[i]),
                      false_positive=int(fp[i]), false_negative=int(fn[i]), true_negative=int(tn[i]),
                      precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i]),
                      specificity=float(specificity[i]), roc_auc_ovr=ranking[i][0],
                      average_precision_ovr=ranking[i][1]) for i in range(classes)]
    confusions = sorted(
        [dict(actual=names[i], predicted=names[j], count=int(matrix[i, j]),
              fraction_of_actual=float(matrix[i, j] / support[i]))
         for i in range(classes) for j in range(classes)
         if i != j and matrix[i, j] > 0],
        key=lambda item: (-item["count"], item["actual"], item["predicted"]))[:10]
    return dict(
        samples=n, correct=correct, errors=n - correct, class_names=names,
        accuracy=accuracy, accuracy_wilson_95_interval=[float(max(0, center - radius)), float(min(1, center + radius))],
        macro_precision=float(precision.mean()), macro_recall=float(recall.mean()), macro_f1=float(f1.mean()),
        weighted_precision=float(np.dot(precision, support) / n),
        weighted_recall=float(np.dot(recall, support) / n), weighted_f1=float(np.dot(f1, support) / n),
        micro_precision=accuracy, micro_recall=accuracy, micro_f1=accuracy,
        balanced_accuracy=float(recall[present].mean()), matthews_correlation_coefficient=float(mcc),
        cohen_kappa=float(kappa), top_k_accuracy=top_k,
        mean_reciprocal_rank=float(np.mean(1 / true_rank)),
        macro_roc_auc_ovr=float(np.mean(valid_auc)) if valid_auc else None,
        macro_average_precision_ovr=float(np.mean(valid_ap)) if valid_ap else None,
        selective_predictions=selective,
        negative_log_likelihood=float(-log_probs[np.arange(n), labels].mean()),
        brier_score=float(brier), expected_calibration_error=float(ece), calibration_bins=calibration_bins,
        mean_confidence=float(confidence.mean()),
        mean_confidence_correct=float(confidence[correct_mask].mean()) if correct else None,
        mean_confidence_wrong=float(confidence[~correct_mask].mean()) if correct < n else None,
        per_class=per_class, confusion_matrix=matrix.tolist(), largest_confusions=confusions,
        reliability_bins=reliability,
        averaging_notes="Macro includes all named classes; zero denominators give 0. Balanced accuracy averages supported classes. Micro recall equals accuracy for single-label multiclass classification.",
        calibration_notes="Softmax of raw logits, without calibration fitted to the test set. ECE uses equal-width confidence bins. Multiclass Brier score sums across classes (range 0..2).",
        interval_notes="Wilson binomial approximation for accuracy; does not measure training-run variation or distribution shift.",
        ranking_notes="One-vs-rest ranking uses softmax probabilities with tied scores grouped. Undefined classes are excluded from ranking macro means and reported as null.",
        selective_notes="Fixed descriptive thresholds; not tuned for deployment. Accepted-set accuracy must be read with coverage and recall counting rejected examples as misses.",
    )


def save_report(metrics, output, predictions_path=None, make_plot=True):
    output = Path(output)
    if output.suffix.lower() != ".json":
        raise ValueError("Report output must have a .json extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(metrics)
    if predictions_path is not None:
        predictions_path = Path(predictions_path)
        result["prediction_file"] = predictions_path.name
        result["prediction_file_sha256"] = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
        result["prediction_array"] = "cpp_logits"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = list(metrics["per_class"][0])
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics["per_class"])
    rows = ["# C++ classifier metrics", "",
            f"Computed from {metrics['samples']:,} saved labeled test predictions; no retraining.", "",
            "| Metric | Value |", "| --- | ---: |"]
    for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "balanced_accuracy"]:
        rows.append(f"| {key.replace('_', ' ').title()} | {metrics[key]:.2%} |")
    for key in ["matthews_correlation_coefficient", "cohen_kappa"]:
        rows.append(f"| {key.replace('_', ' ').title()} | {metrics[key]:.6f} |")
    for k, value in metrics["top_k_accuracy"].items():
        if k != "1":
            rows.append(f"| Top-{k} accuracy | {value:.2%} |")
    for key in ["macro_roc_auc_ovr", "macro_average_precision_ovr", "mean_reciprocal_rank"]:
        value = metrics[key]
        rows.append(f"| {key.replace('_', ' ').title()} | {value:.6f} |" if value is not None else
                    f"| {key.replace('_', ' ').title()} | Undefined |")
    low, high = metrics["accuracy_wilson_95_interval"]
    rows.extend(["", f"Correct: {metrics['correct']:,}; incorrect: {metrics['errors']:,}.",
                 f"Approximate 95% Wilson interval for accuracy: {low:.2%}–{high:.2%}.", "",
                 "Macro recall gives every named class equal weight; micro and weighted recall",
                 "equal accuracy for single-label classification. Macro recall equals accuracy",
                 "when all named classes have equal positive support.", "",
                 "## Per-class recall and F1", "",
                 "| Class | Support | Precision | Recall | F1 |",
                 "| --- | ---: | ---: | ---: | ---: |"])
    for item in metrics["per_class"]:
        rows.append(f"| {item['name']} | {item['support']} | {item['precision']:.2%} | {item['recall']:.2%} | {item['f1']:.2%} |")
    rows.extend(["", "## Most frequent confusions", ""])
    rows.extend(f"- {item['actual']} → {item['predicted']}: {item['count']} images ({item['fraction_of_actual']:.2%} of the true class)."
                for item in metrics["largest_confusions"][:5])
    rows.extend(["", "## Confidence thresholds (descriptive)", "",
                 "| Minimum confidence | Coverage | Accuracy on accepted images | Overall macro recall (rejections count as misses) |",
                 "| --- | ---: | ---: | ---: |"])
    for item in metrics["selective_predictions"]:
        accepted_accuracy = f"{item['accuracy']:.2%}" if item["accuracy"] is not None else "Undefined"
        rows.append(f"| {item['threshold']:.0%} | {item['coverage']:.2%} | {accepted_accuracy} | {item['macro_recall_counting_rejections_as_misses']:.2%} |")
    rows.extend(["", "These fixed thresholds describe the existing predictions. Rejecting examples",
                 "does not improve baseline recall; deployment thresholds must be chosen on",
                 "validation data. ROC-AUC and average precision use one-vs-rest softmax scores."])
    rows.extend(["", "## Confidence diagnostics", "",
                 f"- Negative log-likelihood: {metrics['negative_log_likelihood']:.6f}.",
                 f"- Multiclass Brier score: {metrics['brier_score']:.6f} (sum over classes).",
                 f"- ECE ({metrics['calibration_bins']} equal-width bins): {metrics['expected_calibration_error']:.2%}.",
                 f"- Mean confidence: {metrics['mean_confidence']:.2%}.",
                 "", "These diagnostics use raw softmax probabilities; no calibration or threshold",
                 "was fitted to the test set. Metrics describe this software classifier, not",
                 "bit-accurate RTL or physical FPGA behavior. The interval is a sampling",
                 "approximation and does not cover retraining variability or distribution shift."])
    if make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax, bar) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.2, 1]})
        matrix = np.asarray(metrics["confusion_matrix"])
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
        names = metrics["class_names"]
        ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names,
               xlabel="Predicted class", ylabel="True class", title="Confusion matrix — counts")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if matrix[i, j] > matrix.max() * .55 else "#172033")
        fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
        recalls = np.asarray([item["recall"] for item in metrics["per_class"]]) * 100
        positions = np.arange(len(names))
        bar.barh(positions, recalls, color="#2563eb")
        bar.set(yticks=positions, yticklabels=names, xlim=(0, 105), xlabel="Recall (%)", title="Recall by class")
        bar.invert_yaxis()
        for i, value in enumerate(recalls):
            bar.text(value + .8, i, f"{value:.1f}%", va="center", fontsize=9)
        bar.axvline(metrics["macro_recall"] * 100, color="#b45309", linestyle="--", label="Macro recall")
        bar.legend(loc="lower left")
        fig.suptitle(f"C++ host classifier · {metrics['samples']:,} CIFAR-10 test images · accuracy {metrics['accuracy']:.2%}")
        fig.tight_layout()
        fig.savefig(output.with_suffix(".png"), dpi=160, bbox_inches="tight")
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)
        rows.extend(["", f"![Confusion matrix and class recall]({output.with_suffix('.png').name})"])
    output.with_suffix(".md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=root / "docs/test_predictions.npz")
    parser.add_argument("--output", type=Path, default=root / "docs/classification_metrics.json")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    with np.load(args.predictions, allow_pickle=False) as predictions:
        metrics = classification_metrics(predictions["labels"], predictions["cpp_logits"], CIFAR10_CLASSES)
    save_report(metrics, args.output, args.predictions, make_plot=not args.no_plot)
    print(json.dumps({key: value for key, value in metrics.items()
                      if key not in {"confusion_matrix", "per_class", "largest_confusions", "reliability_bins"}}, indent=2))
