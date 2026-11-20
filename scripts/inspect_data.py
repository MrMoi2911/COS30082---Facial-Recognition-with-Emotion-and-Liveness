import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from longface.data import emotion_datasets, face_datasets
from longface.verification import read_pairs


def main():
    parser = argparse.ArgumentParser(description="Check dataset layout before training")
    parser.add_argument("--task", choices=["face", "emotion"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--verify-images", action="store_true")
    args = parser.parse_args()
    datasets, metadata = (face_datasets(args.data_root) if args.task == "face"
                          else emotion_datasets(args.data_root))
    print(json.dumps(metadata, indent=2))
    for split, dataset in datasets.items():
        counts = Counter(dataset.targets)
        print(f"{split}: {len(dataset)} images, {len(counts)} represented classes")
        if args.verify_images and hasattr(dataset, "samples"):
            for path, _ in dataset.samples:
                with Image.open(path) as image:
                    image.verify()
    if args.task == "face":
        root = Path(args.data_root)
        pairs = read_pairs(root / "verification_pairs_val.txt", root)
        print("Verification pairs:", len(pairs), "Labels:", dict(Counter(pair[2] for pair in pairs)))
        print("Referenced verification images:", len({path for pair in pairs for path in pair[:2]}))
    print("Layout check passed. This does not establish identity independence or absence of near-duplicates.")


if __name__ == "__main__":
    main()
