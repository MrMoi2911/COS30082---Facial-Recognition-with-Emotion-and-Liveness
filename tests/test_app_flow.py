import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from longface.app import AttendanceApp
from longface.liveness import MissingLiveness
from longface.models import face_transform
from longface.storage import AttendanceLog, Gallery


class TextState:
    def set(self, value):
        self.value = value


class FixedEmbedding:
    def encode(self, images):
        return torch.tensor([[1.0, 0.0]])


class AppFlowTests(unittest.TestCase):
    def build_app(self, directory, demo=False):
        app = AttendanceApp.__new__(AttendanceApp)
        app.args = argparse.Namespace(demo=demo, metric="cosine", registration_samples=3,
                                      registration_consistency=0.65)
        app.device = torch.device("cpu")
        app.face_model = FixedEmbedding()
        app.face_transform = face_transform(64)
        app.emotion_model = None
        app.liveness = MissingLiveness()
        app.quality_config = {}
        app.gallery = Gallery(Path(directory) / "gallery.json", "fixture", 2)
        app.attendance = AttendanceLog(Path(directory) / "attendance.csv", "fixture")
        app.threshold = 0.8
        app.status, app.notice = TextState(), TextState()
        app.stable_id, app.stable_count = None, 0
        app.pending, app.previous_box = None, None
        app.last_sample_time = 0.0
        return app

    def test_normal_flow_does_not_log_when_liveness_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self.build_app(temporary)
            app.gallery.register("person", "Fixture", [[1, 0]])
            frame = np.random.default_rng(8).integers(60, 190, (160, 160, 3), dtype=np.uint8)
            for _ in range(5):
                app.process_face(frame, (20, 20, 120, 120))
            self.assertFalse(app.attendance.path.exists())

    def test_demo_registration_then_daily_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = self.build_app(temporary, demo=True)
            frame = np.random.default_rng(8).integers(60, 190, (160, 160, 3), dtype=np.uint8)
            app.pending = {"person_id": "person", "name": "Fixture", "vectors": [], "started": 99.0}
            for now in [100.0, 101.0, 102.0]:
                with patch("longface.app.time.monotonic", return_value=now):
                    app.process_face(frame, (20, 20, 120, 120))
            self.assertIsNone(app.pending)
            self.assertIn("person", app.gallery.people)
            for _ in range(4):
                app.process_face(frame, (20, 20, 120, 120))
            lines = app.attendance.path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn(",unavailable,demo,", lines[1])


if __name__ == "__main__":
    unittest.main()
