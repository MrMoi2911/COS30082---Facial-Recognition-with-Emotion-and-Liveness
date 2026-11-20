from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def read_pairs(path, root):
    root = Path(root).resolve()
    pairs, seen = [], {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 3 or fields[2] not in {"0", "1"}:
            raise ValueError(f"Line {number}: expected image1 image2 label(0/1)")
        images = []
        for relative in fields[:2]:
            target = (root / relative).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Line {number}: image path outside data root")
            images.append(target)
        key = tuple(sorted(str(image) for image in images))
        label = int(fields[2])
        if key in seen and seen[key] != label:
            raise ValueError(f"Conflicting labels for pair at line {number}")
        seen[key] = label
        pairs.append((images[0], images[1], label))
    if not pairs or {pair[2] for pair in pairs} != {0, 1}:
        raise ValueError("ROC evaluation requires both same-person and different-person pairs")
    missing = sorted({str(image) for pair in pairs for image in pair[:2] if not image.is_file()})
    if missing:
        raise FileNotFoundError(f"{len(missing)} referenced images missing. Examples: {missing[:5]}")
    return pairs


def pair_scores(pairs, embeddings):
    first = np.stack([embeddings[str(pair[0])] for pair in pairs])
    second = np.stack([embeddings[str(pair[1])] for pair in pairs])
    first = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    second = second / np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-12)
    return {"cosine": np.sum(first * second, axis=1),
            "negative_euclidean": -np.linalg.norm(first - second, axis=1)}


def choose_threshold(labels, scores, target_fpr=0.01):
    if not 0 <= target_fpr <= 1:
        raise ValueError("target_fpr must be in [0, 1]")
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    candidates = np.flatnonzero((fpr <= target_fpr) & np.isfinite(thresholds))
    if len(candidates):
        index = int(candidates[np.argmax(tpr[candidates])])
        threshold = float(thresholds[index])
    else:
        threshold = float(np.max(scores) + 1e-6)
    return threshold


def score_summary(labels, scores, threshold):
    labels = np.asarray(labels)
    accepted = np.asarray(scores) >= threshold
    return {"roc_auc": float(roc_auc_score(labels, scores)), "threshold": float(threshold),
            "tpr": float(accepted[labels == 1].mean()),
            "fpr": float(accepted[labels == 0].mean()),
            "accuracy": float((accepted == labels).mean())}
