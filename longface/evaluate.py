import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score, roc_curve)
from torch.utils.data import DataLoader, Dataset

from .common import choose_device, file_sha256, write_json
from .data import emotion_datasets, face_datasets
from .models import face_transform, load_model
from .verification import choose_threshold, pair_scores, read_pairs, score_summary


class PairImages(Dataset):
    def __init__(self, paths, image_size):
        self.paths = list(paths)
        self.transform = face_transform(image_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), str(self.paths[index])


def plot_history(run, output):
    history = Path(run) / "history.csv"
    if not history.is_file():
        return
    frame = pd.read_csv(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for prefix, label in [("train", "Train"), ("val", "Validation")]:
        axes[0].plot(frame.epoch, frame[f"{prefix}_loss"], label=label)
        axes[1].plot(frame.epoch, frame[f"{prefix}_accuracy"], label=label)
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(Path(output) / "training_curves.png", dpi=160)
    plt.close(fig)


def evaluate_classification(args):
    device = choose_device(args.device)
    model, config = load_model(args.checkpoint, device)
    if config["task"] == "face":
        datasets, _ = face_datasets(args.data_root, config["image_size"])
    else:
        datasets, _ = emotion_datasets(args.data_root, config["seed"], config["val_fraction"])
    dataset = datasets[args.split]
    if dataset.classes != config["classes"]:
        raise ValueError("Dataset class order differs from the trained model")
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    actual, predicted = [], []
    with torch.inference_mode():
        for images, labels in loader:
            predicted.extend(model(images.to(device)).argmax(1).cpu().tolist())
            actual.extend(labels.tolist())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = {"task": config["task"], "split": args.split, "samples": len(actual),
               "accuracy": float(accuracy_score(actual, predicted)),
               "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
               "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
               "checkpoint_sha256": file_sha256(args.checkpoint)}
    write_json(output / "classification_metrics.json", metrics)
    report = classification_report(actual, predicted, labels=list(range(len(dataset.classes))),
                                   target_names=dataset.classes, output_dict=True, zero_division=0)
    write_json(output / "classification_report.json", report)
    pd.DataFrame({"true_label": actual, "predicted_label": predicted}).to_csv(output / "predictions.csv", index=False)
    if len(dataset.classes) <= 20:
        fig, ax = plt.subplots(figsize=(8, 7))
        ConfusionMatrixDisplay.from_predictions(actual, predicted,
                                                labels=list(range(len(dataset.classes))),
                                                display_labels=dataset.classes,
                                                xticks_rotation=45, ax=ax, colorbar=False)
        ax.set_title(f"{config['task']} / {args.split}")
        fig.tight_layout()
        fig.savefig(output / "confusion_matrix.png", dpi=160)
        plt.close(fig)
    plot_history(Path(args.checkpoint).parent, output)
    print(metrics)


def evaluate_verification(args):
    device = choose_device(args.device)
    model, config = load_model(args.checkpoint, device)
    if config["task"] != "face":
        raise ValueError("Verification requires a face embedding model")
    evaluation_pairs = read_pairs(args.pairs, args.data_root)
    calibration_pairs = read_pairs(args.calibration_pairs, args.data_root) if args.calibration_pairs else evaluation_pairs
    if args.calibration_pairs:
        calibration_images = {str(path) for pair in calibration_pairs for path in pair[:2]}
        evaluation_images = {str(path) for pair in evaluation_pairs for path in pair[:2]}
        if calibration_images & evaluation_images:
            raise ValueError("Calibration and evaluation share image paths. Use disjoint sets or omit --calibration-pairs.")
    paths = sorted({path for pair in evaluation_pairs + calibration_pairs for path in pair[:2]})
    loader = DataLoader(PairImages(paths, config["image_size"]), batch_size=args.batch_size,
                        num_workers=args.workers)
    embeddings = {}
    with torch.inference_mode():
        for images, names in loader:
            vectors = model.encode(images.to(device)).cpu().numpy()
            embeddings.update(zip(names, vectors))
    scores = pair_scores(evaluation_pairs, embeddings)
    calibration_scores = pair_scores(calibration_pairs, embeddings)
    labels = [pair[2] for pair in evaluation_pairs]
    calibration_labels = [pair[2] for pair in calibration_pairs]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = file_sha256(args.checkpoint)
    scope = "separate image-disjoint calibration; identity independence not established" if args.calibration_pairs else "validation only; threshold selected on these same pairs"
    metrics, thresholds = {}, {}
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, values in scores.items():
        threshold = choose_threshold(calibration_labels, calibration_scores[name], args.target_fpr)
        metrics[name] = score_summary(labels, values, threshold)
        thresholds[name] = threshold
        fpr, tpr, _ = roc_curve(labels, values)
        ax.plot(fpr, tpr, label=f"{name}: AUC={metrics[name]['roc_auc']:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Face verification ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "roc.png", dpi=160)
    plt.close(fig)
    result = {"scope": scope, "pairs": len(labels), "target_calibration_fpr": args.target_fpr,
              "checkpoint_sha256": fingerprint, "pairs_sha256": file_sha256(args.pairs),
              "metrics": metrics,
              "note": "Pairwise thresholds require separate validation for 1:N gallery matching. No guaranteed webcam FPR."}
    write_json(output / "verification_metrics.json", result)
    write_json(output / "thresholds.json", {"format_version": 1, "checkpoint_sha256": fingerprint,
                                            "score_rule": "accept if score >= threshold",
                                            "scope": scope, "thresholds": thresholds})
    pd.DataFrame({"image1": [str(pair[0]) for pair in evaluation_pairs],
                  "image2": [str(pair[1]) for pair in evaluation_pairs],
                  "label": labels, **scores}).to_csv(output / "pair_scores.csv", index=False)
    print(result)


def parser():
    result = argparse.ArgumentParser(description="Evaluate classification or pairwise verification")
    result.add_argument("mode", choices=["classification", "verification"])
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--data-root", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--split", choices=["val", "test"], default="test")
    result.add_argument("--pairs")
    result.add_argument("--calibration-pairs")
    result.add_argument("--target-fpr", type=float, default=0.01)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--workers", type=int, default=2)
    result.add_argument("--device", default="auto")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.mode == "verification":
        if not arguments.pairs:
            raise ValueError("--pairs is required for verification")
        evaluate_verification(arguments)
    else:
        evaluate_classification(arguments)
