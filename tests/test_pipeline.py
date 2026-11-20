import argparse
import ast
import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import nbformat
import numpy as np
from PIL import Image
import torch

from longface.evaluate import evaluate_classification, evaluate_verification
from longface.models import FaceClassifier, load_model
from longface.train import parser as train_parser, train
from scripts.colab_helpers import safe_extract
from scripts.export_run import export_run


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_face_train_resume_verify_and_export_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            random = np.random.default_rng(42)
            for split in ["train_data", "val_data", "test_data"]:
                for label in ["id_a", "id_b"]:
                    directory = data / "classification_data" / split / label
                    directory.mkdir(parents=True)
                    for index in range(2):
                        Image.fromarray(random.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(directory / f"{index}.png")
            verification = data / "verification_data"
            verification.mkdir()
            for index in range(4):
                Image.fromarray(random.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(verification / f"{index}.png")
            pairs = data / "verification_pairs_val.txt"
            pairs.write_text("verification_data/0.png verification_data/1.png 1\n"
                             "verification_data/0.png verification_data/2.png 0\n"
                             "verification_data/1.png verification_data/3.png 0\n")
            run = root / "face_classification"
            options = ["--task", "face", "--data-root", str(data), "--output", str(run),
                       "--epochs", "1", "--batch-size", "2", "--workers", "0", "--device", "cpu",
                       "--from-scratch", "--train-backbone", "all", "--image-size", "64"]
            with contextlib.redirect_stdout(io.StringIO()):
                train(train_parser().parse_args(options))
                options[options.index("--epochs") + 1] = "2"
                train(train_parser().parse_args(options + ["--resume"]))
            last = torch.load(run / "last.pt", weights_only=True)
            self.assertEqual(last["epoch"], 2)
            self.assertEqual(len(last["history"]), 2)
            model, config = load_model(run / "best.pt")
            with torch.inference_mode():
                embeddings = model.encode(torch.randn(2, 3, 64, 64))
            self.assertTrue(torch.allclose(embeddings.norm(dim=1), torch.ones(2), atol=1e-5))
            args = argparse.Namespace(checkpoint=str(run / "best.pt"), data_root=str(data),
                                      split="test", batch_size=2, workers=0, device="cpu",
                                      output=str(run / "classification"))
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate_classification(args)
                args.pairs, args.calibration_pairs, args.target_fpr = str(pairs), None, 0.01
                args.output = str(run / "verification")
                evaluate_verification(args)
                destination = export_run(run, root / "results.zip")
            metrics = json.loads((run / "verification" / "verification_metrics.json").read_text())
            self.assertIn("validation only", metrics["scope"])
            self.assertEqual(set(metrics["metrics"]), {"cosine", "negative_euclidean"})
            with zipfile.ZipFile(destination) as archive:
                self.assertIn("models/face_classification/best.pt", archive.namelist())
                self.assertIn("reports/face_classification/verification/thresholds.json", archive.namelist())
                self.assertFalse(any(name.endswith("last.pt") for name in archive.namelist()))

    def test_emotion_train_and_test_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            random = np.random.default_rng(23)
            for split in ["train", "val", "test"]:
                for label in ["happy", "sad"]:
                    directory = root / "data" / split / label
                    directory.mkdir(parents=True)
                    for index in range(2):
                        Image.fromarray(random.integers(0, 255, (48, 48), dtype=np.uint8)).save(directory / f"{index}.png")
            output = root / "emotion"
            options = ["--task", "emotion", "--data-root", str(root / "data"), "--output", str(output),
                       "--epochs", "1", "--batch-size", "2", "--workers", "0", "--device", "cpu", "--class-weights"]
            with contextlib.redirect_stdout(io.StringIO()):
                train(train_parser().parse_args(options))
                evaluate_classification(argparse.Namespace(checkpoint=str(output / "best.pt"),
                    data_root=str(root / "data"), split="test", batch_size=2, workers=0,
                    device="cpu", output=str(output / "classification")))
            metrics = json.loads((output / "classification" / "classification_metrics.json").read_text())
            self.assertEqual(metrics["samples"], 4)
            self.assertTrue((output / "classification" / "confusion_matrix.png").is_file())

    def test_notebooks_have_valid_python_and_no_fabricated_outputs(self):
        root = Path(__file__).resolve().parents[1]
        for path in (root / "notebooks").glob("*.ipynb"):
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    ast.parse(cell.source)
                    self.assertEqual(cell.outputs, [])
                    self.assertIsNone(cell.execution_count)

    def test_zip_cannot_escape_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "unsafe.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("../escape.txt", "bad")
            with self.assertRaises(ValueError):
                safe_extract(source, root / "destination")
            self.assertFalse((root / "escape.txt").exists())

    def test_default_finetuning_freezes_earlier_batchnorm(self):
        model = FaceClassifier(2, pretrained=False, train_backbone="layer4")
        model.train()
        initial = model.backbone.bn1.running_mean.clone()
        self.assertFalse(model.backbone.conv1.weight.requires_grad)
        self.assertTrue(model.backbone.layer4[0].conv1.weight.requires_grad)
        model(torch.randn(2, 3, 64, 64)).sum().backward()
        self.assertTrue(torch.equal(initial, model.backbone.bn1.running_mean))
        self.assertIsNotNone(model.embedding.weight.grad)


if __name__ == "__main__":
    unittest.main()
