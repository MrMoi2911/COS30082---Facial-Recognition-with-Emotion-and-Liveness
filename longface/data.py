from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from .models import emotion_transform, face_transform


FER_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class FolderSamples(Dataset):
    def __init__(self, samples, classes, transform):
        self.samples = list(samples)
        self.classes = list(classes)
        self.targets = [int(label) for _, label in self.samples]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


def folder_samples(directory, classes=None):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing image directory: {directory}")
    folder = ImageFolder(directory)
    classes = folder.classes if classes is None else list(classes)
    mapping = {name: index for index, name in enumerate(classes)}
    unknown = set(folder.classes) - set(classes)
    if unknown:
        raise ValueError(f"Unknown labels in {directory}: {sorted(unknown)}")
    samples = [(path, mapping[folder.classes[label]]) for path, label in folder.samples]
    return samples, classes


def face_datasets(root, image_size=224):
    root = Path(root)
    base = root / "classification_data" if (root / "classification_data").is_dir() else root
    train, classes = folder_samples(base / "train_data")
    val, _ = folder_samples(base / "val_data", classes)
    test, _ = folder_samples(base / "test_data", classes)
    datasets = {
        "train": FolderSamples(train, classes, face_transform(image_size, True)),
        "val": FolderSamples(val, classes, face_transform(image_size)),
        "test": FolderSamples(test, classes, face_transform(image_size)),
    }
    return datasets, {"source": str(base.resolve()), "split_policy": "provided train_data/val_data/test_data"}


class FERPixels(Dataset):
    def __init__(self, frame, transform):
        self.transform = transform
        self.classes = FER_CLASSES.copy()
        self.targets = frame["emotion"].astype(int).tolist()
        self.row_ids = frame.index.astype(int).tolist()
        self.images = []
        for index, pixels in zip(frame.index, frame["pixels"]):
            array = np.fromstring(str(pixels), sep=" ", dtype=np.int16)
            if len(array) != 2304 or np.any(array < 0) or np.any(array > 255):
                raise ValueError(f"FER2013 row {index} must contain 2304 pixels in [0,255]")
            self.images.append(array.astype(np.uint8).reshape(48, 48))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.transform(Image.fromarray(self.images[index])), self.targets[index]


def emotion_datasets(source, seed=42, val_fraction=0.15):
    source = Path(source)
    if source.is_file():
        frame = pd.read_csv(source)
        if not {"emotion", "pixels", "Usage"}.issubset(frame.columns):
            raise ValueError("Expected FER2013 CSV with emotion, pixels, and Usage columns")
        if frame["emotion"].isna().any() or not frame["emotion"].isin(range(7)).all():
            raise ValueError("FER2013 labels must be integers from 0 through 6")
        usage = {"train": "Training", "val": "PublicTest", "test": "PrivateTest"}
        if not frame["Usage"].isin(usage.values()).all():
            raise ValueError("Unknown or missing FER2013 Usage value")
        datasets = {}
        for split, name in usage.items():
            selected = frame[frame["Usage"] == name]
            if selected.empty:
                raise ValueError(f"Missing FER2013 split {name}")
            datasets[split] = FERPixels(selected, emotion_transform(split == "train"))
        metadata = {"source": str(source.resolve()), "split_policy": usage}
        return datasets, metadata
    train, classes = folder_samples(source / "train")
    test, _ = folder_samples(source / "test", classes)
    val_path = source / "val"
    if not val_path.is_dir() and (source / "validation").is_dir():
        val_path = source / "validation"
    if val_path.is_dir():
        val, _ = folder_samples(val_path, classes)
        policy = "provided train/val(or validation)/test"
    else:
        train_indices, val_indices = train_test_split(
            np.arange(len(train)), test_size=val_fraction,
            stratify=[label for _, label in train], random_state=seed)
        original = train
        train = [original[int(i)] for i in train_indices]
        val = [original[int(i)] for i in val_indices]
        policy = f"stratified {val_fraction:.0%} validation from train only; test untouched; seed={seed}"
    datasets = {name: FolderSamples(samples, classes, emotion_transform(name == "train"))
                for name, samples in [("train", train), ("val", val), ("test", test)]}
    return datasets, {"source": str(source.resolve()), "split_policy": policy}


def save_manifest(datasets, output):
    rows = []
    for split, dataset in datasets.items():
        if hasattr(dataset, "samples"):
            for path, label in dataset.samples:
                rows.append({"split": split, "sample": str(path), "label": label,
                             "class_name": dataset.classes[label]})
        else:
            for row_id, label in zip(dataset.row_ids, dataset.targets):
                rows.append({"split": split, "sample": f"csv_row:{row_id}", "label": label,
                             "class_name": dataset.classes[label]})
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(output) / "split_manifest.csv", index=False)
    return frame.groupby(["split", "class_name"]).size().unstack(fill_value=0)
