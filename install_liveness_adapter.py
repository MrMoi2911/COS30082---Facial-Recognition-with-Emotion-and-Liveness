"""Install the adapter for the model trained in 04_liveness.ipynb.

Place this file in the project root and run:
    python3 install_liveness_adapter.py

The installer adds longface/liveness_adapter.py and changes only the
liveness input from the face crop to the RGB frame used by the notebook.
"""

import ast
from pathlib import Path
import shlex


ADAPTER_SOURCE = '''"""ResNet18 adapter matching 04_liveness.ipynb.

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
'''

OLD_CALL = "live = checked_prediction(self.liveness, crop)"
NEW_CALL = "live = checked_prediction(self.liveness, rgb)"


def install(root):
    root = Path(root)
    app_path = root / "longface" / "app.py"
    adapter_path = root / "longface" / "liveness_adapter.py"
    backup_path = app_path.with_name("app.py.before_liveness_adapter.bak")
    original = app_path.read_bytes()
    source = original.decode("utf-8")

    if source.count(OLD_CALL) == 1 and NEW_CALL not in source:
        updated = source.replace(OLD_CALL, NEW_CALL, 1)
        needs_patch = True
    elif source.count(NEW_CALL) == 1 and OLD_CALL not in source:
        updated = source
        needs_patch = False
    else:
        raise RuntimeError(
            "The liveness call differs from the supplied app.py. "
            "No files were changed. Share the current longface/app.py for review."
        )

    ast.parse(updated, filename=str(app_path))
    ast.parse(ADAPTER_SOURCE, filename=str(adapter_path))
    if adapter_path.exists() and adapter_path.read_text(encoding="utf-8") != ADAPTER_SOURCE:
        raise FileExistsError(f"An existing adapter differs: {adapter_path}. No files were changed.")
    if needs_patch and backup_path.exists() and backup_path.read_bytes() != original:
        raise FileExistsError(f"A different app backup already exists: {backup_path}. No files were changed.")

    if needs_patch and not backup_path.exists():
        backup_path.write_bytes(original)
    adapter_path.write_text(ADAPTER_SOURCE, encoding="utf-8")
    if needs_patch:
        app_path.write_bytes(updated.encode("utf-8"))

    print("Installed: longface/liveness_adapter.py")
    print("Liveness uses the full RGB frame. Recognition and emotion still use the face crop.")
    if needs_patch:
        print("Backup: longface/app.py.before_liveness_adapter.bak")
    print("\nRun from the project root:")
    print(shlex.join([
        "python3", "-m", "longface.app",
        "--face-model", "runs/face_classification/best.pt",
        "--thresholds", "runs/face_classification/verification/thresholds.json",
        "--emotion-model", "runs/emotion/best.pt",
        "--liveness", "longface.liveness_adapter:create_liveness",
        "--liveness-checkpoint", "runs/liveness/best_liveness_model.pth",
        "--camera", "0",
    ]))


def main():
    candidates = [Path.cwd(), Path(__file__).resolve().parent]
    root = next((p for p in candidates if (p / "longface" / "app.py").is_file()), None)
    if root is None:
        raise SystemExit("Run this installer from the project root containing longface/app.py.")
    try:
        install(root)
    except (OSError, RuntimeError, UnicodeError, SyntaxError) as error:
        raise SystemExit(f"Installation stopped: {error}") from error


if __name__ == "__main__":
    main()
