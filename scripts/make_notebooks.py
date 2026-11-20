"""Regenerate the two Colab notebooks from their source cells."""

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def shared_setup():
    return [
        markdown("""
## 1. Prepare Drive and GPU
Choose **Runtime > Change runtime type > GPU**. Upload the source ZIP to
`MyDrive/Long_Face_Project/Long_Facial_Attendance.zip`. Upload the relevant dataset ZIP
to that same Drive folder. Change the paths below if your names differ.
Data is extracted to local Colab disk. Checkpoints are saved to Drive after each completed epoch.
"""),
        code("""
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
DRIVE_ROOT = Path('/content/drive/MyDrive/Long_Face_Project')
CODE_ZIP = DRIVE_ROOT / 'Long_Facial_Attendance.zip'
"""),
        markdown("## 2. Load the shared source code"),
        code("""
import os
import stat
import sys
import zipfile
import subprocess

code_destination = Path('/content/long_source').resolve()
code_destination.mkdir(parents=True, exist_ok=True)
if not CODE_ZIP.is_file():
    raise FileNotFoundError(f'Upload the source ZIP first: {CODE_ZIP}')
with zipfile.ZipFile(CODE_ZIP) as archive:
    for entry in archive.infolist():
        target = (code_destination / entry.filename).resolve()
        if not target.is_relative_to(code_destination) or stat.S_ISLNK(entry.external_attr >> 16):
            raise ValueError(f'Unsafe archive entry: {entry.filename}')
    archive.extractall(code_destination)

REPO = code_destination / 'Long_Facial_Attendance'
if not (REPO / 'longface' / 'train.py').is_file():
    raise FileNotFoundError('The source ZIP should contain the Long_Facial_Attendance folder')
os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-colab.txt'], check=True)
try:
    import torch
    import torchvision
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision'], check=True)
    import torch
    import torchvision
print('PyTorch:', torch.__version__, '| torchvision:', torchvision.__version__)
print('GPU available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    print('Full training may be slow. Select a GPU runtime before continuing.')

def run_module(module, *arguments):
    subprocess.run([sys.executable, '-m', module, *map(str, arguments)], check=True)
"""),
    ]


def export_cells(task):
    return [
        markdown("""
## Export to your local Git repository
The ZIP contains `models/` and `reports/` folders. Extract it into the shared repo root.
Weights stay outside Git through `.gitignore`; share their real Drive link with your teammate.
Keep `last.pt` on Drive for resuming. Also download this executed notebook using
**File > Download > Download .ipynb**, and save it to the corresponding file in `notebooks/`.
Never put Kaggle credentials or personal registration data in notebook output.
"""),
        code(f"""
from scripts.export_run import export_run
EXPORT = DRIVE_ROOT / 'exports' / '{task}_results.zip'
export_run(RUN, EXPORT)
print('Saved to:', EXPORT)

# Set True when you are ready to download the result ZIP.
DOWNLOAD_RESULTS = False
if DOWNLOAD_RESULTS:
    from google.colab import files
    files.download(str(EXPORT))
"""),
    ]


def make_face():
    cells = [markdown("""
# Face verification: classification-based embeddings
Owner: Long. Train ResNet18 with identity classification and cross-entropy loss.
Fine-tune the final residual block and the new embedding/classification layers.
At inference, remove the identity head and compare L2-normalised 128-dimensional embeddings.

This notebook does not contain trained results. It requires the actual Kaggle face images.
It implements one required baseline. The teammate implements metric learning separately.
""")]
    cells += shared_setup()
    cells += [
        markdown("## 3. Set dataset and training options"),
        code("""
FACE_ZIP = DRIVE_ROOT / 'face_dataset.zip'
DATA_DIRECTORY = Path('/content/datasets/face')
RUN = DRIVE_ROOT / 'runs' / 'face_classification'
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SEED = 42
RESUME = False
VERIFY_ALL_IMAGES = False
# If Colab runs out of GPU memory, start a new run with BATCH_SIZE = 32.
"""),
        markdown("""
## 4. Extract and inspect the dataset
The root must contain `classification_data/train_data`, `val_data`, `test_data`,
`verification_data`, and `verification_pairs_val.txt`. Identity labels come from folder names.
The official splits are kept unchanged. Set `VERIFY_ALL_IMAGES=True` for a full image decode check.
"""),
        code("""
from scripts.colab_helpers import safe_extract, find_face_root
if not DATA_DIRECTORY.exists() or not any(DATA_DIRECTORY.iterdir()):
    safe_extract(FACE_ZIP, DATA_DIRECTORY)
FACE_ROOT = find_face_root(DATA_DIRECTORY)
print('Dataset root:', FACE_ROOT)
inspection = [sys.executable, 'scripts/inspect_data.py', '--task', 'face', '--data-root', str(FACE_ROOT)]
if VERIFY_ALL_IMAGES:
    inspection.append('--verify-images')
subprocess.run(inspection, check=True)
"""),
        code("""
from longface.data import face_datasets
from PIL import Image
import matplotlib.pyplot as plt

datasets, metadata = face_datasets(FACE_ROOT)
fig, axes = plt.subplots(1, 6, figsize=(12, 3))
samples = datasets['train'].samples
for axis, (path, label) in zip(axes, samples[::max(1, len(samples)//6)][:6]):
    with Image.open(path) as image:
        axis.imshow(image.convert('RGB'))
    axis.set_title(datasets['train'].classes[label])
    axis.axis('off')
plt.tight_layout()
plt.show()
print('Check face crops, pose, and image quality before training.')
"""),
        markdown("""
## 5. Train and save to Drive
`best.pt` is selected by validation classification accuracy. `last.pt` includes optimiser,
scheduler, RNG states, and training history. Test data is not used for early stopping.
After a runtime disconnect, rerun setup/extraction, keep the same settings, and set `RESUME=True`.
`EPOCHS` is the total target epoch count, not the number of extra epochs.
The loader checks the saved split manifest, but it does not hash every image's contents.
"""),
        code("""
arguments = ['--task', 'face', '--data-root', FACE_ROOT, '--output', RUN,
             '--epochs', EPOCHS, '--batch-size', BATCH_SIZE, '--lr', LEARNING_RATE,
             '--seed', SEED, '--embedding-dim', 128, '--image-size', 224,
             '--train-backbone', 'layer4', '--patience', 5, '--workers', 2]
if RESUME:
    arguments.append('--resume')
run_module('longface.train', *arguments)
with (RUN / 'environment.txt').open('w') as stream:
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
"""),
        markdown("## 6. Evaluate classification on the provided test split"),
        code("""
run_module('longface.evaluate', 'classification', '--checkpoint', RUN / 'best.pt',
           '--data-root', FACE_ROOT, '--split', 'test', '--output', RUN / 'classification')
"""),
        markdown("""
## 7. Evaluate face verification
Compute ROC-AUC from `verification_pairs_val.txt`, as required in the brief.
Evaluate cosine similarity and negative Euclidean distance so that higher scores always mean
more similar. The embeddings are L2-normalised; these metrics have equivalent rankings apart
from numerical ties, so similar AUC values are expected.

This notebook also selects a demonstration threshold on these same validation pairs at a
target empirical FPR of 1%. This is not an independent test and does not guarantee a webcam
false-accept rate. A 1:N registration gallery needs its own validation before real deployment.
"""),
        code("""
run_module('longface.evaluate', 'verification', '--checkpoint', RUN / 'best.pt',
           '--data-root', FACE_ROOT, '--pairs', FACE_ROOT / 'verification_pairs_val.txt',
           '--target-fpr', 0.01, '--output', RUN / 'verification')
"""),
        code("""
import json
import pandas as pd
from IPython.display import display, Image as DisplayImage
display(pd.read_csv(RUN / 'history.csv'))
print(json.dumps(json.loads((RUN / 'verification' / 'verification_metrics.json').read_text()), indent=2))
display(DisplayImage(filename=str(RUN / 'verification' / 'roc.png')))
display(DisplayImage(filename=str(RUN / 'classification' / 'training_curves.png')))
"""),
        markdown("""
## 8. Record the real findings
Complete these notes after running, using the saved results:
- Best validation epoch and test classification accuracy.
- Verification AUC for each distance metric.
- Threshold selection procedure and its limitations.
- Failure cases involving pose, blur, lighting, and unknown identities.
- Comparison with the teammate's metric-learning baseline using the same pairs and preprocessing.
Do not fill in scores before training. Do not claim identity-independent testing without checking identities.
"""),
    ]
    cells += export_cells("face_classification")
    return cells


def make_emotion():
    cells = [markdown("""
# Facial expression classification
Owner: Long. Train a small CNN from scratch using FER2013 labels.
Inputs are grayscale 48x48 face images. The saved checkpoint contains the actual label order.
The predictions describe visible expression classes, not a diagnosis of a person's mental state.

This notebook contains no trained results. Download a public FER2013 dataset separately.
Supported formats: the original `fer2013.csv`, or folders with `train/` and `test/`.
""")]
    cells += shared_setup()
    cells += [
        markdown("## 3. Set dataset and training options"),
        code("""
EMOTION_ZIP = DRIVE_ROOT / 'emotion_dataset.zip'
DATA_DIRECTORY = Path('/content/datasets/emotion')
RUN = DRIVE_ROOT / 'runs' / 'emotion'
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SEED = 42
RESUME = False
"""),
        markdown("""
## 4. Prepare splits and inspect labels
For the original CSV, `Training`, `PublicTest`, and `PrivateTest` become train, validation,
and test. For folders, a provided `val/` or `validation/` folder is used. If absent, 15% of
the training images are reserved for validation using a stratified split with seed 42.
The test folder is never used to create validation data. Do not mix the original FER2013
labels with FER+ vote labels unless you implement the corresponding label conversion.
"""),
        code("""
from scripts.colab_helpers import safe_extract, find_emotion_source
if not DATA_DIRECTORY.exists() or not any(DATA_DIRECTORY.iterdir()):
    safe_extract(EMOTION_ZIP, DATA_DIRECTORY)
EMOTION_SOURCE = find_emotion_source(DATA_DIRECTORY)
print('Emotion source:', EMOTION_SOURCE)
subprocess.run([sys.executable, 'scripts/inspect_data.py', '--task', 'emotion',
                '--data-root', str(EMOTION_SOURCE)], check=True)
"""),
        code("""
from longface.data import emotion_datasets
from collections import Counter
import matplotlib.pyplot as plt
datasets, metadata = emotion_datasets(EMOTION_SOURCE, seed=SEED)
print('Labels:', datasets['train'].classes)
counts = Counter(datasets['train'].targets)
plt.figure(figsize=(8, 3))
plt.bar(datasets['train'].classes, [counts[index] for index in range(len(datasets['train'].classes))])
plt.ylabel('Training images')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 6, figsize=(12, 3))
for index, axis in enumerate(axes):
    image, label = datasets['val'][index % len(datasets['val'])]
    axis.imshow((image[0] * 0.5 + 0.5).numpy(), cmap='gray', vmin=0, vmax=1)
    axis.set_title(datasets['val'].classes[label])
    axis.axis('off')
plt.tight_layout()
plt.show()
"""),
        markdown("""
## 5. Train the expression model
Use cross-entropy with weights calculated from training labels only.
Choose `best.pt` using validation macro-F1, not test accuracy. Augmentation applies only to train.
The checkpoint is saved after each completed epoch. Set `RESUME=True` after a disconnect.
Keep the dataset, seed, batch size, and other settings unchanged when resuming.
"""),
        code("""
arguments = ['--task', 'emotion', '--data-root', EMOTION_SOURCE, '--output', RUN,
             '--epochs', EPOCHS, '--batch-size', BATCH_SIZE, '--lr', LEARNING_RATE,
             '--seed', SEED, '--patience', 7, '--class-weights', '--workers', 2]
if RESUME:
    arguments.append('--resume')
run_module('longface.train', *arguments)
with (RUN / 'environment.txt').open('w') as stream:
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
"""),
        markdown("## 6. Evaluate once on the held-out test split"),
        code("""
run_module('longface.evaluate', 'classification', '--checkpoint', RUN / 'best.pt',
           '--data-root', EMOTION_SOURCE, '--split', 'test', '--output', RUN / 'classification')
"""),
        code("""
import json
import pandas as pd
from IPython.display import display, Image as DisplayImage
display(pd.read_csv(RUN / 'history.csv'))
print(json.dumps(json.loads((RUN / 'classification' / 'classification_metrics.json').read_text()), indent=2))
display(DisplayImage(filename=str(RUN / 'classification' / 'confusion_matrix.png')))
display(DisplayImage(filename=str(RUN / 'classification' / 'training_curves.png')))
"""),
        markdown("""
## 7. Record the real findings
- Test accuracy, balanced accuracy, and macro-F1.
- Per-class precision/recall/F1 and the main confusions.
- Effects of class imbalance and image quality.
- Webcam examples that succeed or fail after local integration.
- Dataset source, actual split sizes, and whether duplicates or subject overlap were checked.
No performance guarantee is made before the experiment is run.
"""),
    ]
    cells += export_cells("emotion")
    return cells


def main():
    for filename, cells in [("01_classification.ipynb", make_face()), ("03_emotion.ipynb", make_emotion())]:
        notebook = nbf.v4.new_notebook(cells=cells)
        notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}, "colab": {"name": filename}, "accelerator": "GPU"}
        nbf.validate(notebook)
        nbf.write(notebook, ROOT / "notebooks" / filename)
        print(filename)


if __name__ == "__main__":
    main()
