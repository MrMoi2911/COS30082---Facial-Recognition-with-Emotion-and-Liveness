import shutil
import stat
import zipfile
from pathlib import Path


def safe_extract(archive, destination):
    archive, destination = Path(archive), Path(destination).resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Upload this ZIP to Drive first: {archive}")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as stream:
        members = stream.infolist()
        for member in members:
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Unsafe ZIP entry: {member.filename}")
        required = sum(member.file_size for member in members)
        if required > shutil.disk_usage(destination).free:
            raise OSError("Not enough local Colab disk space to extract this dataset")
        stream.extractall(destination)
    return destination


def find_face_root(directory):
    directory = Path(directory)
    candidates = [path.parent for path in directory.rglob("classification_data")
                  if path.is_dir() and (path / "train_data").is_dir()]
    candidates = [path for path in candidates if (path / "verification_pairs_val.txt").is_file()]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one face dataset root; found {candidates}. Check the extracted ZIP.")
    return candidates[0]


def find_emotion_source(directory):
    directory = Path(directory)
    csvs = list(directory.rglob("fer2013.csv"))
    if len(csvs) == 1:
        return csvs[0]
    folders = [path.parent for path in directory.rglob("train")
               if path.is_dir() and (path.parent / "test").is_dir()]
    if len(folders) == 1:
        return folders[0]
    raise ValueError("Expected one fer2013.csv or one root containing train/ and test/. Set the source explicitly if needed.")
