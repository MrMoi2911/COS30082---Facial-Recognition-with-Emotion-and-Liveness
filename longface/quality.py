from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class QualityResult:
    accepted: bool
    reason: str
    blur: float
    brightness: float


def check_quality(rgb, min_size=100, min_blur=80.0, min_brightness=45.0,
                  max_brightness=210.0):
    if rgb is None or rgb.size == 0:
        return QualityResult(False, "No face crop", 0.0, 0.0)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    reasons = []
    if min(rgb.shape[:2]) < min_size:
        reasons.append("Move closer")
    if blur < min_blur:
        reasons.append("Image blurry; hold still")
    if brightness < min_brightness:
        reasons.append("Too dark")
    if brightness > max_brightness:
        reasons.append("Too bright")
    return QualityResult(not reasons, "; ".join(reasons) if reasons else "Quality accepted", blur, brightness)


def box_iou(first, second):
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return intersection / max(aw * ah + bw * bh - intersection, 1)
