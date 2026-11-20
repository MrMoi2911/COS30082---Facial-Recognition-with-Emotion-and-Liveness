import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from PIL import Image
from longface.common import write_json
from longface.quality import check_quality


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", default="configs/quality.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    settings = json.loads(Path(args.config).read_text())
    rows = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    for category in ["acceptable", "blurred", "dark", "bright"]:
        for path in sorted((Path(args.root) / category).glob("*")):
            if path.suffix.lower() not in extensions:
                continue
            with Image.open(path) as image:
                result = check_quality(np.array(image.convert("RGB")), **settings)
            rows.append({"file": path.name, "category": category, "accepted": result.accepted,
                         "blur": result.blur, "brightness": result.brightness, "reason": result.reason})
    if not rows:
        raise ValueError("No labelled face crops found. See docs/QUALITY_EXPERIMENT.md")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "quality_results.csv", index=False)
    result = {}
    for category, group in frame.groupby("category"):
        result[category] = {"samples": len(group), "acceptance_rate": float(group.accepted.mean()),
                            "rejection_rate": float(1 - group.accepted.mean())}
    write_json(output / "quality_metrics.json", {"settings": settings, "by_category": result})
    print(result)


if __name__ == "__main__":
    main()
