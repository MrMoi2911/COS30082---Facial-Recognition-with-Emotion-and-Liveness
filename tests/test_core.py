import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from longface.data import emotion_datasets
from longface.liveness import (LivenessResult, MissingLiveness, allow_attendance,
                               checked_prediction)
from longface.quality import check_quality
from longface.storage import AttendanceLog, Gallery
from longface.verification import choose_threshold, pair_scores, read_pairs, score_summary


class CoreTests(unittest.TestCase):
    def test_pairs_validate_labels_missing_files_and_conflicts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ["a.jpg", "b.jpg", "c.jpg"]:
                Image.new("RGB", (8, 8)).save(root / name)
            pairs = root / "pairs.txt"
            pairs.write_text("a.jpg b.jpg 1\na.jpg c.jpg 0\n")
            self.assertEqual(len(read_pairs(pairs, root)), 2)
            pairs.write_text("a.jpg b.jpg 1\nb.jpg a.jpg 0\n")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                read_pairs(pairs, root)
            pairs.write_text("a.jpg b.jpg 1\na.jpg absent.jpg 0\n")
            with self.assertRaises(FileNotFoundError):
                read_pairs(pairs, root)

    def test_similarity_direction_and_calibration_fpr(self):
        pairs = [(Path("a"), Path("b"), 1), (Path("a"), Path("c"), 0)]
        embeddings = {"a": np.array([1., 0.]), "b": np.array([0.9, 0.1]),
                      "c": np.array([0., 1.])}
        for scores in pair_scores(pairs, embeddings).values():
            self.assertGreater(scores[0], scores[1])
            threshold = choose_threshold([1, 0], scores, target_fpr=0)
            summary = score_summary([1, 0], scores, threshold)
            self.assertEqual(summary["roc_auc"], 1)
            self.assertEqual(summary["fpr"], 0)

    def test_constant_scores_can_reject_all(self):
        threshold = choose_threshold([0, 1], [0.3, 0.3], target_fpr=0)
        self.assertGreater(threshold, 0.3)
        self.assertEqual(score_summary([0, 1], [0.3, 0.3], threshold)["fpr"], 0)

    def test_missing_or_failing_liveness_blocks_normal_attendance(self):
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        missing = checked_prediction(MissingLiveness(), rgb)
        self.assertFalse(allow_attendance(missing))
        self.assertTrue(allow_attendance(missing, demo=True))
        for status in ["spoof", "uncertain"]:
            self.assertFalse(allow_attendance(LivenessResult(status), demo=True))

        class Broken:
            def predict(self, face):
                raise RuntimeError("model failed")

        self.assertFalse(allow_attendance(checked_prediction(Broken(), rgb)))

        class Invalid:
            def predict(self, face):
                return LivenessResult("live", float("nan"))

        self.assertFalse(allow_attendance(checked_prediction(Invalid(), rgb)))

    def test_gallery_matches_unknown_and_rejects_changed_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gallery.json"
            gallery = Gallery(path, "checkpoint-a", 2)
            gallery.register("long", "Long", [[1., 0.], [0.99, 0.01]])
            self.assertEqual(gallery.match([1, 0], "cosine", 0.8)[0], "long")
            self.assertIsNone(gallery.match([0, 1], "cosine", 0.8)[0])
            reloaded = Gallery(path, "checkpoint-a", 2)
            self.assertEqual(reloaded.match([1, 0], "negative_euclidean", -0.1)[0], "long")
            with self.assertRaisesRegex(ValueError, "different"):
                Gallery(path, "checkpoint-b", 2)

    def test_attendance_dedup_survives_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "attendance.csv"
            now = datetime(2026, 11, 18, 10, tzinfo=timezone(timedelta(hours=7)))
            log = AttendanceLog(path, "model")
            self.assertTrue(log.record("long", "Long", 0.9, "live", now=now))
            self.assertFalse(AttendanceLog(path, "model").record("long", "Long", 0.9, "live", now=now))
            self.assertTrue(AttendanceLog(path, "model").record("long", "Long", 0.9, "live", now=now + timedelta(days=1)))

    def test_quality_rejects_bad_crops_and_accepts_sharp_fixture(self):
        dark = np.zeros((120, 120, 3), dtype=np.uint8)
        self.assertFalse(check_quality(dark).accepted)
        sharp = np.random.default_rng(42).integers(60, 190, (120, 120, 3), dtype=np.uint8)
        self.assertTrue(check_quality(sharp).accepted)
        self.assertFalse(check_quality(sharp[:30, :30]).accepted)

    def test_emotion_validation_comes_from_train_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split, size in [("train", 10), ("test", 3)]:
                for label in ["happy", "sad"]:
                    directory = root / split / label
                    directory.mkdir(parents=True)
                    for index in range(size):
                        Image.new("RGB", (48, 48), color=(index, 100, 100)).save(directory / f"{index}.png")
            datasets, _ = emotion_datasets(root)
            train = {path for path, _ in datasets["train"].samples}
            val = {path for path, _ in datasets["val"].samples}
            test = {path for path, _ in datasets["test"].samples}
            self.assertFalse(train & val or train & test or val & test)
            self.assertTrue(all("/train/" in path for path in val))
            self.assertEqual(len(train) + len(val), 20)

    def test_original_fer_csv_preserves_splits_and_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "fer2013.csv"
            pixels = " ".join(["100"] * 2304)
            pd.DataFrame({"emotion": [0, 3, 6], "pixels": [pixels] * 3,
                          "Usage": ["Training", "PublicTest", "PrivateTest"]}).to_csv(source, index=False)
            datasets, _ = emotion_datasets(source)
            self.assertEqual(datasets["train"].classes[0], "angry")
            self.assertEqual(datasets["val"].targets, [3])
            self.assertEqual(datasets["test"].targets, [6])


if __name__ == "__main__":
    unittest.main()
