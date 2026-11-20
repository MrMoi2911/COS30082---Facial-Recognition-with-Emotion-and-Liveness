import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .common import atomic_torch_save, choose_device, file_sha256, runtime_versions, seed_everything, write_json
from .data import emotion_datasets, face_datasets, save_manifest
from .models import build_model


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss = 0.0
    actual, predicted = [], []
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False, desc="train" if optimizer else "validation"):
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            actual.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(1).detach().cpu().tolist())
    return {"loss": total_loss / len(actual), "accuracy": float(accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0))}


def train(args):
    seed_everything(args.seed)
    device = choose_device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    resume_path = output / "last.pt"
    if resume_path.exists() and not args.resume:
        raise FileExistsError("last.pt exists. Use --resume or choose a new output folder.")
    if args.resume and not resume_path.exists():
        raise FileNotFoundError(f"No checkpoint to resume: {resume_path}")
    if args.task == "face":
        datasets, metadata = face_datasets(args.data_root, args.image_size)
    else:
        datasets, metadata = emotion_datasets(args.data_root, args.seed, args.val_fraction)
    classes = datasets["train"].classes
    if len(classes) < 2:
        raise ValueError("Training requires at least two classes")
    config = {"task": args.task, "architecture": "resnet18_embedding" if args.task == "face" else "emotion_cnn",
              "classes": classes, "embedding_dim": args.embedding_dim,
              "train_backbone": args.train_backbone, "image_size": args.image_size if args.task == "face" else 48,
              "seed": args.seed, "val_fraction": args.val_fraction,
              "initial_weights": "random" if args.from_scratch or args.task == "emotion" else "ImageNet",
              "preprocessing": "rgb_imagenet_resize" if args.task == "face" else "gray48_normalize_0.5"}
    settings = {"batch_size": args.batch_size, "lr": args.lr, "weight_decay": args.weight_decay,
                "class_weights": args.class_weights, "patience": args.patience}
    model = build_model(config, pretrained=not args.from_scratch and not args.resume).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    generator = torch.Generator().manual_seed(args.seed)
    best, start_epoch, bad_epochs, history = -1.0, 0, 0, []
    state = None
    if args.resume:
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
        if state["config"] != config or state["training_settings"] != settings:
            raise ValueError("Resume settings differ. Restore the original settings or start a new run.")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        best, start_epoch = state["best_score"], state["epoch"]
        bad_epochs, history = state["bad_epochs"], state["history"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        numpy_state = state["numpy_rng"]
        np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32),
                             numpy_state[2], numpy_state[3], numpy_state[4]))
        generator.set_state(state["loader_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    if start_epoch >= args.epochs:
        print(f"Already completed {start_epoch} epochs. Increase --epochs to continue.")
        return
    manifest_path = output / "split_manifest.csv"
    if state is not None and manifest_path.exists():
        previous_manifest_hash = file_sha256(manifest_path)
        if previous_manifest_hash != state["manifest_sha256"]:
            raise ValueError("The saved split manifest was modified")
    counts = save_manifest(datasets, output)
    manifest_hash = file_sha256(manifest_path)
    if state is not None and manifest_hash != state["manifest_sha256"]:
        raise ValueError("Dataset paths, labels, or splits changed since the checkpoint")
    counts.to_csv(output / "class_counts.csv")
    weights = None
    if args.class_weights:
        frequencies = np.bincount(datasets["train"].targets, minlength=len(classes))
        if np.any(frequencies == 0):
            raise ValueError("Every class needs training examples")
        weights = torch.tensor(len(datasets["train"]) / (len(classes) * frequencies),
                               dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train",
                                num_workers=args.workers, pin_memory=device.type == "cuda",
                                generator=generator if split == "train" else None)
               for split, dataset in datasets.items() if split != "test"}
    metadata.update({"counts": {key: len(value) for key, value in datasets.items()},
                     "versions": runtime_versions(), "selection_metric": "val_accuracy" if args.task == "face" else "val_macro_f1",
                     "device": str(device), "manifest_sha256": manifest_hash})
    write_json(output / "config.json", config)
    write_json(output / "run_metadata.json", metadata)
    write_json(output / "training_settings.json", {**settings, "target_epochs": args.epochs, "workers": args.workers})
    print(f"Device: {device}. Classes: {len(classes)}. Split sizes: {metadata['counts']}")
    for epoch in range(start_epoch, args.epochs):
        training = run_epoch(model, loaders["train"], criterion, device, optimizer)
        validation = run_epoch(model, loaders["val"], criterion, device)
        score = validation["accuracy" if args.task == "face" else "macro_f1"]
        improved = score > best
        bad_epochs = 0 if improved else bad_epochs + 1
        best = max(best, score)
        scheduler.step(score)
        row = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"],
               **{f"train_{key}": value for key, value in training.items()},
               **{f"val_{key}": value for key, value in validation.items()}}
        history.append(row)
        inference = {"format_version": 1, "config": config, "model_state": model.state_dict(),
                     "epoch": epoch + 1, "best_score": best, "training_settings": settings}
        if improved:
            atomic_torch_save(inference, output / "best.pt")
        numpy_state = np.random.get_state()
        checkpoint = {**inference, "optimizer_state": optimizer.state_dict(),
                      "scheduler_state": scheduler.state_dict(), "bad_epochs": bad_epochs,
                      "history": history, "manifest_sha256": manifest_hash,
                      "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                      "numpy_rng": (numpy_state[0], numpy_state[1].tolist(), numpy_state[2],
                                    numpy_state[3], numpy_state[4]),
                      "loader_rng": generator.get_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        atomic_torch_save(checkpoint, resume_path)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(json.dumps(row))
        if bad_epochs >= args.patience:
            print(f"Early stopping. Best validation score: {best:.4f}")
            break
    print(f"Saved model: {output / 'best.pt'}")


def parser():
    result = argparse.ArgumentParser(description="Train Long's face classification or emotion model")
    result.add_argument("--task", choices=["face", "emotion"], required=True)
    result.add_argument("--data-root", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--epochs", type=int, default=15)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--lr", type=float, default=0.001)
    result.add_argument("--weight-decay", type=float, default=0.0001)
    result.add_argument("--embedding-dim", type=int, default=128)
    result.add_argument("--image-size", type=int, default=224)
    result.add_argument("--train-backbone", choices=["layer4", "all"], default="layer4")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--workers", type=int, default=2)
    result.add_argument("--patience", type=int, default=5)
    result.add_argument("--val-fraction", type=float, default=0.15)
    result.add_argument("--device", default="auto")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--from-scratch", action="store_true")
    result.add_argument("--class-weights", action="store_true")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.epochs < 1 or arguments.batch_size < 1 or arguments.patience < 1:
        raise ValueError("epochs, batch-size, and patience must be positive")
    train(arguments)
