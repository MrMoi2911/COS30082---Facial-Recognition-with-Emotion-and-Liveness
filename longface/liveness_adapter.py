"""ResNet18 adapter matching 04_liveness.ipynb.

Input: a full RGB uint8 webcam frame, not a cropped face.
Classes: 0 = Fake, 1 = Real. Decision threshold: 0.5.
The notebook's video-level results average five frames. This adapter makes
one frame-level prediction per call; webcam performance needs validation.
"""

from pathlib import Path
import math

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms

from .liveness import LivenessResult


class NotebookLivenessAdapter:
    def __init__(self, checkpoint_path):
        if not checkpoint_path:
            raise ValueError("Pass --liveness-checkpoint runs/liveness/best_liveness_model.pth")
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Liveness checkpoint not found: {path}")

        # Load only the weights saved by the liveness notebook.
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model.load_state_dict(state, strict=True)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def predict(self, frame_rgb):
        if (
            not isinstance(frame_rgb, np.ndarray)
            or frame_rgb.dtype != np.uint8
            or frame_rgb.ndim != 3
            or frame_rgb.shape[2] != 3
            or frame_rgb.shape[0] == 0
            or frame_rgb.shape[1] == 0
        ):
            return LivenessResult("unavailable", None, "Expected a non-empty RGB uint8 frame")

        # Match the notebook's OpenCV resize and ImageNet normalization.
        resized = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        image = Image.fromarray(resized).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            score = float(torch.softmax(logits, dim=1)[0, 1].item())

        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            return LivenessResult("uncertain", None, "Model returned an invalid probability")

        status = "live" if score >= 0.5 else "spoof"
        return LivenessResult(status, score, f"ResNet18: P(real)={score:.3f}")


def create_liveness(checkpoint_path):
    return NotebookLivenessAdapter(checkpoint_path)
