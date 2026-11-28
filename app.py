import argparse
import csv
import cv2
from dataclasses import dataclass
from datetime import datetime
import hashlib
import importlib
import json
import math
import numpy as np
from pathlib import Path
from PIL import Image, ImageTk
import re
import time
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms, transforms as T
from torchvision.models import ResNet18_Weights, resnet18


def choose_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FaceClassifier(nn.Module):
    def __init__(self, num_classes, embedding_dim=128, pretrained=False,
                 train_backbone="layer4"):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        self.embedding = nn.Linear(512, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        if train_backbone not in {"all", "layer4"}:
            raise ValueError("train_backbone must be all or layer4")
        self.train_backbone = train_backbone
        if train_backbone == "layer4":
            for name, parameter in self.backbone.named_parameters():
                parameter.requires_grad = name.startswith("layer4.")

    def train(self, mode=True):
        super().train(mode)
        if mode and self.train_backbone == "layer4":
            self.backbone.eval()
            self.backbone.layer4.train()
        return self

    def encode(self, images):
        return F.normalize(self.embedding(self.backbone(images)), p=2, dim=1)

    def forward(self, images):
        features = self.embedding(self.backbone(images))
        return self.classifier(features)


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        layers = []
        channels = [1, 32, 64, 128]
        for source, target in zip(channels, channels[1:]):
            layers.extend([nn.Conv2d(source, target, 3, padding=1),
                           nn.BatchNorm2d(target), nn.ReLU(inplace=True), nn.MaxPool2d(2)])
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((3, 3)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 3 * 3, 128),
                                        nn.ReLU(inplace=True), nn.Dropout(0.4),
                                        nn.Linear(128, num_classes))

    def forward(self, images):
        return self.classifier(self.features(images))


def face_transform(image_size=224, training=False):
    operations = [T.Resize((image_size, image_size))]
    if training:
        operations.extend([T.RandomHorizontalFlip(),
                           T.ColorJitter(brightness=0.15, contrast=0.15)])
    operations.extend([T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return T.Compose(operations)


def emotion_transform(training=False):
    operations = [T.Grayscale(num_output_channels=1), T.Resize((48, 48))]
    if training:
        operations.extend([T.RandomHorizontalFlip(), T.RandomRotation(8)])
    operations.extend([T.ToTensor(), T.Normalize([0.5], [0.5])])
    return T.Compose(operations)


def build_model(config, pretrained=False):
    if config["task"] == "face":
        return FaceClassifier(len(config["classes"]), config["embedding_dim"],
                              pretrained, config["train_backbone"])
    if config["task"] == "emotion":
        return EmotionCNN(len(config["classes"]))
    raise ValueError("Unknown task in model config")


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format. Expected format_version 1.")
    model = build_model(checkpoint["config"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint["config"]


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


def normalise(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or not np.isfinite(vector).all() or norm < 1e-8:
        raise ValueError("Invalid embedding")
    return vector / norm


class Gallery:
    def __init__(self, path, model_sha256, embedding_dim):
        self.path = Path(path)
        self.model_sha256 = model_sha256
        self.embedding_dim = embedding_dim
        self.people = {}
        if self.path.is_file():
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if value["model_sha256"] != model_sha256:
                raise ValueError("Gallery uses a different face checkpoint. Re-register with the selected model.")
            for person_id, person in value["people"].items():
                vector = normalise(person["embedding"])
                if len(vector) != embedding_dim:
                    raise ValueError("Gallery embedding dimension differs from model")
                self.people[person_id] = {**person, "embedding": vector.tolist()}

    def register(self, person_id, name, embeddings):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", person_id):
            raise ValueError("ID must contain 1-48 letters, numbers, underscores, or hyphens")
        if not name.strip():
            raise ValueError("Name cannot be empty")
        vectors = [normalise(vector) for vector in embeddings]
        if not vectors or any(len(vector) != self.embedding_dim for vector in vectors):
            raise ValueError("Invalid registration embeddings")
        template = normalise(np.mean(vectors, axis=0))
        self.people[person_id] = {"name": name.strip(), "embedding": template.tolist(),
                                  "samples": len(vectors), "registered_at": datetime.now().astimezone().isoformat()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"model_sha256": self.model_sha256, "people": self.people},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def match(self, embedding, metric, threshold):
        if not self.people:
            return None, "Unknown", None
        vector = normalise(embedding)
        identities = list(self.people)
        templates = np.array([self.people[key]["embedding"] for key in identities], dtype=np.float32)
        if metric == "cosine":
            scores = templates @ vector
        elif metric == "negative_euclidean":
            scores = -np.linalg.norm(templates - vector, axis=1)
        else:
            raise ValueError("Unknown similarity metric")
        index = int(np.argmax(scores))
        score = float(scores[index])
        person_id = identities[index]
        if score < threshold:
            return None, "Unknown", score
        return person_id, self.people[person_id]["name"], score


class AttendanceLog:
    fields = ["timestamp", "person_id", "name", "score", "liveness", "mode", "model_sha256"]

    def __init__(self, path, model_sha256):
        self.path = Path(path)
        self.model_sha256 = model_sha256
        self.seen = set()
        if self.path.is_file():
            with self.path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    self.seen.add((row["person_id"], row["timestamp"][:10]))

    def record(self, person_id, name, score, liveness, mode="normal", now=None):
        now = now or datetime.now().astimezone()
        key = (person_id, now.date().isoformat())
        if key in self.seen:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields)
            if not exists:
                writer.writeheader()
            writer.writerow({"timestamp": now.isoformat(), "person_id": person_id,
                             "name": name, "score": f"{score:.5f}", "liveness": liveness,
                             "mode": mode, "model_sha256": self.model_sha256})
        self.seen.add(key)
        return True


@dataclass
class LivenessResult:
    status: str
    score: float | None = None
    detail: str = ""


class MissingLiveness:
    def predict(self, face_rgb):
        return LivenessResult("unavailable", None, "Teammate's trained liveness model is not connected")


def load_liveness(spec=None, checkpoint=None):
    if spec in {"builtin", "longface.liveness_adapter:create_liveness"}:
        return create_liveness(checkpoint)
    if not spec:
        return MissingLiveness()
    module_name, separator, factory_name = spec.partition(":")
    if not separator:
        raise ValueError("Use --liveness package.module:create_liveness")
    factory = getattr(importlib.import_module(module_name), factory_name)
    return factory(checkpoint)


def checked_prediction(adapter, face_rgb):
    try:
        result = adapter.predict(face_rgb)
        if not isinstance(result, LivenessResult):
            raise TypeError("Adapter must return longface.liveness.LivenessResult")
        if result.status not in {"live", "spoof", "uncertain", "unavailable"}:
            raise ValueError("Invalid liveness status")
        if result.status == "live" and (result.score is None or not math.isfinite(result.score)
                                         or not 0 <= result.score <= 1):
            raise ValueError("Live result requires a finite probability in [0,1]")
        return result
    except Exception as error:
        return LivenessResult("unavailable", None, f"Liveness error: {error}")


def allow_attendance(result, demo=False):
    return result.status == "live" or (demo and result.status == "unavailable")


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



def draw_face_label(rgb, box, emotion):
    x, y, width, height = map(int, box)
    green = (50, 220, 80)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, rgb.shape[1] / 1150)
    thickness = max(1, round(scale * 2))
    (tw, th), base = cv2.getTextSize(emotion, font, scale, thickness)
    label_height = th + base + 8
    left = max(0, min(x, rgb.shape[1] - tw - 9))
    top = y - label_height if y >= label_height else min(y + 2, rgb.shape[0] - label_height)
    top = max(0, top)
    cv2.rectangle(rgb, (x, y), (x + width, y + height), green, 2)
    cv2.rectangle(rgb, (left, top), (left + tw + 8, top + label_height), green, -1)
    cv2.putText(rgb, emotion, (left + 4, top + th + 4), font, scale,
                (10, 20, 12), thickness, cv2.LINE_AA)


class AttendanceApp:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.device = choose_device(args.device)
        self.face_model, face_config = load_model(args.face_model, self.device)
        if face_config["task"] != "face":
            raise ValueError("--face-model must be a trained face embedding checkpoint")
        self.face_transform = face_transform(face_config["image_size"])
        self.fingerprint = file_sha256(args.face_model)
        thresholds = json.loads(Path(args.thresholds).read_text())
        if thresholds["checkpoint_sha256"] != self.fingerprint:
            raise ValueError("Thresholds were computed using a different checkpoint")
        self.threshold = float(thresholds["thresholds"][args.metric])
        self.emotion_model = None
        self.emotion_config = None
        if args.emotion_model:
            self.emotion_model, self.emotion_config = load_model(args.emotion_model, self.device)
            if self.emotion_config["task"] != "emotion":
                raise ValueError("--emotion-model must be an emotion checkpoint")
        self.emotion_transform = emotion_transform()
        self.liveness = load_liveness(args.liveness, args.liveness_checkpoint)
        state = Path(args.state_dir) / ("demo" if args.demo else "normal")
        self.gallery = Gallery(state / "gallery.json", self.fingerprint, face_config["embedding_dim"])
        self.attendance = AttendanceLog(state / "attendance.csv", self.fingerprint)
        self.quality_config = json.loads(Path(args.quality_config).read_text()) if args.quality_config else {}
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(str(cascade_path))
        if self.detector.empty():
            raise RuntimeError("OpenCV face detector could not be loaded")
        self.capture = cv2.VideoCapture(args.camera)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError("Cannot open webcam. Check camera index and camera permission.")
        self.pending = None
        self.previous_box = None
        self.last_sample_time = 0.0
        self.stable_id, self.stable_count = None, 0
        self.frame_count = 0
        self.failed_reads = 0
        self.closed = False
        root.title("Long | Face Attendance")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("q", lambda event: self.close())
        root.bind("<Escape>", lambda event: self.close())
        mode = "DEMO: NO LIVENESS GUARANTEE. Separate demo records." if args.demo else "NORMAL: Attendance requires trained liveness to report LIVE."
        ttk.Label(root, text=mode, foreground="#a04400", wraplength=850).pack(padx=12, pady=8)
        self.video = ttk.Label(root)
        self.video.pack(padx=12)
        self.status = tk.StringVar(value="Stand in front of the camera")
        ttk.Label(root, textvariable=self.status, wraplength=850).pack(padx=12, pady=8)
        self.notice = tk.StringVar(value=f"Registered people: {len(self.gallery.people)}. Unknown: use Register.")
        ttk.Label(root, textvariable=self.notice, wraplength=850).pack(padx=12, pady=4)
        controls = ttk.Frame(root)
        controls.pack(pady=10)
        ttk.Button(controls, text="Register new person", command=self.begin_registration).pack(side="left", padx=5)
        ttk.Button(controls, text="Cancel registration", command=self.cancel_registration).pack(side="left", padx=5)
        ttk.Button(controls, text="Quit", command=self.close).pack(side="left", padx=5)
        self.root.after(10, self.update_frame)

    def begin_registration(self):
        person_id = simpledialog.askstring("Registration", "Person ID (letters, numbers, _ or -):", parent=self.root)
        if not person_id:
            return
        import re
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", person_id):
            messagebox.showerror("Invalid ID", "Use 1-48 letters, numbers, underscores, or hyphens.")
            return
        if person_id in self.gallery.people:
            if not messagebox.askyesno("Replace registration?", "This ID exists. Replace its face template?"):
                return
        name = simpledialog.askstring("Registration", "Display name:", parent=self.root)
        if not name or not name.strip():
            return
        self.pending = {"person_id": person_id, "name": name.strip(), "vectors": [], "started": time.monotonic()}
        self.previous_box = None
        self.last_sample_time = 0.0
        self.stable_id, self.stable_count = None, 0
        self.notice.set("Registration started. Keep one face visible and move slightly between samples.")

    def cancel_registration(self):
        self.pending = None
        self.previous_box = None
        self.notice.set("Registration cancelled")

    @staticmethod
    def face_crop(rgb, box):
        x, y, width, height = [int(value) for value in box]
        # Keep a small border around the detected face.
        margin_x, margin_y = int(width * 0.1), int(height * 0.1)
        return rgb[max(0, y - margin_y):min(rgb.shape[0], y + height + margin_y),
                   max(0, x - margin_x):min(rgb.shape[1], x + width + margin_x)]

    def predict_expression(self, crop):
        if self.emotion_model is None:
            return "unavailable", None
        image = self.emotion_transform(Image.fromarray(crop)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            confidence, index = self.emotion_model(image).softmax(1)[0].max(0)
        return self.emotion_config["classes"][int(index)], float(confidence)

    def process_face(self, rgb, box, crop=None, emotion=None):
        if crop is None:
            crop = self.face_crop(rgb, box)
        if emotion is None:
            emotion = self.predict_expression(crop)
        quality = check_quality(crop, **self.quality_config)
        live = checked_prediction(self.liveness, rgb)
        with torch.inference_mode():
            image = self.face_transform(Image.fromarray(crop)).unsqueeze(0).to(self.device)
            vector = self.face_model.encode(image)[0].cpu().numpy()
        permitted = allow_attendance(live, self.args.demo)
        person_id, name, score = self.gallery.match(vector, self.args.metric, self.threshold)
        label, confidence = emotion
        detail = label if confidence is None else f"{label} ({confidence:.0%})"
        self.status.set(f"Identity: {name} | Expression: {detail} | Liveness: {live.status}\n"
                        f"{quality.reason} | Sharpness: {quality.blur:.1f} | Brightness: {quality.brightness:.1f}\n{live.detail}")
        now = time.monotonic()
        if self.pending is not None:
            if now - self.pending["started"] > 45:
                self.cancel_registration()
                self.notice.set("Registration timed out. Correct lighting/liveness, then try again.")
                return name
            if self.previous_box is not None and box_iou(self.previous_box, box) < 0.25:
                self.pending["vectors"].clear()
                self.notice.set("Face position changed abruptly. Registration samples reset.")
            self.previous_box = tuple(int(value) for value in box)
            vectors = self.pending["vectors"]
            consistent = not vectors or float(normalise(vectors[0]) @ normalise(vector)) >= self.args.registration_consistency
            if quality.accepted and permitted and consistent and now - self.last_sample_time >= 0.5:
                vectors.append(vector)
                self.last_sample_time = now
                self.notice.set(f"Registration: {len(vectors)}/{self.args.registration_samples} accepted samples")
                if len(vectors) >= self.args.registration_samples:
                    self.gallery.register(self.pending["person_id"], self.pending["name"], vectors)
                    self.notice.set(f"Registered {self.pending['name']}. Total: {len(self.gallery.people)}")
                    self.pending = None
                    self.previous_box = None
            elif not permitted:
                self.notice.set(f"Registration blocked: liveness is {live.status}")
            elif not consistent:
                self.notice.set("Face differs from the first sample. Cancel and register again if needed.")
            return name
        if not permitted or not quality.accepted or person_id is None:
            self.stable_id, self.stable_count = None, 0
            return name
        if person_id == self.stable_id:
            self.stable_count += 1
        else:
            self.stable_id, self.stable_count = person_id, 1
        if self.stable_count >= 3:
            recorded = self.attendance.record(person_id, name, score, live.status,
                                               "demo" if self.args.demo else "normal")
            self.notice.set(f"Attendance recorded: {name}" if recorded else f"Already recorded today: {name}")
        return name

    def update_frame(self):
        if self.closed:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.failed_reads += 1
            if self.failed_reads >= 30:
                messagebox.showerror("Camera", "Camera stopped returning frames")
                self.close()
                return
            self.root.after(100, self.update_frame)
            return
        self.failed_reads = 0
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = self.detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                               scaleFactor=1.1, minNeighbors=5, minSize=(70, 70))
        self.frame_count += 1
        if len(boxes) == 1:
            box = boxes[0]
            try:
                crop = self.face_crop(rgb, box)
                emotion = self.predict_expression(crop)
                if self.frame_count % self.args.infer_every == 0:
                    self.process_face(rgb, box, crop, emotion)
                label, _ = emotion
                draw_face_label(rgb, box, label)
            except Exception as error:
                self.stable_id, self.stable_count = None, 0
                self.status.set(f"Processing error: {error}")
        else:
            self.stable_id, self.stable_count = None, 0
            if self.pending is not None:
                self.pending["vectors"].clear()
                self.previous_box = None
                if time.monotonic() - self.pending["started"] > 45:
                    self.cancel_registration()
            self.status.set("No face detected" if len(boxes) == 0 else "Multiple faces. Keep one person in view.")
        preview = Image.fromarray(rgb)
        preview.thumbnail((900, 600))
        self.photo = ImageTk.PhotoImage(preview)
        self.video.configure(image=self.photo)
        self.root.after(20, self.update_frame)

    def close(self):
        if not self.closed:
            self.closed = True
            self.capture.release()
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Run the local webcam app after training on Colab")
    parser.add_argument("--face-model", default="runs/face_classification/best.pt")
    parser.add_argument("--thresholds", default="runs/face_classification/verification/thresholds.json")
    emotion_path = Path("runs/emotion/best.pt")
    alternate = Path("runs/emotion_no_weights/best.pt")
    if not emotion_path.is_file() and alternate.is_file():
        emotion_path = alternate
    parser.add_argument("--emotion-model", default=str(emotion_path))
    parser.add_argument("--metric", choices=["cosine", "negative_euclidean"], default="cosine")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--state-dir", default="runtime")
    parser.add_argument("--quality-config", default="configs/quality.json" if Path("configs/quality.json").is_file() else None)
    parser.add_argument("--liveness", default="builtin")
    parser.add_argument("--liveness-checkpoint", default="runs/liveness/best_liveness_model.pth")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--registration-samples", type=int, default=10)
    parser.add_argument("--registration-consistency", type=float, default=0.65)
    parser.add_argument("--infer-every", type=int, default=3)
    args = parser.parse_args()
    if args.registration_samples < 2 or args.infer_every < 1:
        parser.error("registration-samples must be >=2; infer-every must be >=1")
    root = tk.Tk()
    try:
        AttendanceApp(root, args)
    except Exception:
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
