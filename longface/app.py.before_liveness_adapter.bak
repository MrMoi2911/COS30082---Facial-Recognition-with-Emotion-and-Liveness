import argparse
import json
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk

from .common import choose_device, file_sha256
from .liveness import allow_attendance, checked_prediction, load_liveness
from .models import emotion_transform, face_transform, load_model
from .quality import box_iou, check_quality
from .storage import AttendanceLog, Gallery, normalise


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

    def process_face(self, rgb, box):
        x, y, width, height = [int(value) for value in box]
        # Keep a small border around the detected face.
        margin_x, margin_y = int(width * 0.1), int(height * 0.1)
        crop = rgb[max(0, y - margin_y):min(rgb.shape[0], y + height + margin_y),
                   max(0, x - margin_x):min(rgb.shape[1], x + width + margin_x)]
        quality = check_quality(crop, **self.quality_config)
        live = checked_prediction(self.liveness, crop)
        with torch.inference_mode():
            vector = self.face_model.encode(self.face_transform(Image.fromarray(crop)).unsqueeze(0).to(self.device))[0].cpu().numpy()
            emotion = "not loaded"
            if self.emotion_model is not None:
                logits = self.emotion_model(self.emotion_transform(Image.fromarray(crop)).unsqueeze(0).to(self.device))
                probabilities = logits.softmax(1)[0]
                confidence, index = probabilities.max(0)
                emotion = f"{self.emotion_config['classes'][int(index)]} ({float(confidence):.0%})"
        permitted = allow_attendance(live, self.args.demo)
        person_id, name, score = self.gallery.match(vector, self.args.metric, self.threshold)
        self.status.set(f"Identity: {name} | Expression: {emotion} | Liveness: {live.status}\n"
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
            x, y, width, height = [int(value) for value in boxes[0]]
            cv2.rectangle(rgb, (x, y), (x + width, y + height), (50, 220, 80), 2)
            if self.frame_count % self.args.infer_every == 0:
                try:
                    # Use the original RGB image, without the overlay rectangle.
                    self.process_face(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), boxes[0])
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
    parser.add_argument("--face-model", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--emotion-model")
    parser.add_argument("--metric", choices=["cosine", "negative_euclidean"], default="cosine")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--state-dir", default="runtime")
    parser.add_argument("--quality-config", default="configs/quality.json")
    parser.add_argument("--liveness")
    parser.add_argument("--liveness-checkpoint")
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
