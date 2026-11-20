import argparse
import json
import zipfile
from pathlib import Path


def export_run(run, destination):
    run, destination = Path(run), Path(destination)
    if not (run / "best.pt").is_file():
        raise FileNotFoundError("No best.pt found. Finish training before exporting.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_names = {"best.pt", "config.json", "run_metadata.json", "training_settings.json"}
    report_names = {"history.csv", "class_counts.csv", "split_manifest.csv", "environment.txt"}
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(run.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(run)
            if path.name in checkpoint_names and len(relative.parts) == 1:
                target = Path("models") / run.name / path.name
            elif path.name in report_names and len(relative.parts) == 1:
                target = Path("reports") / run.name / path.name
            elif relative.parts[0] in {"classification", "verification"}:
                target = Path("reports") / run.name / relative
            else:
                continue
            bundle.write(path, target.as_posix())
        bundle.writestr("EXPORT_README.txt", "Extract into the shared repo root. models/ is ignored by Git. "
                        "Keep last.pt on Drive for resuming. Download the executed notebook separately from Colab.\n")
    print(json.dumps({"export": str(destination), "bytes": destination.stat().st_size}))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    export_run(args.run, args.destination)
