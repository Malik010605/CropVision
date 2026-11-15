# CropVision setup and Phase 1–2 workflow

The commands below target macOS Apple Silicon and Python 3.11. They do not
assume CUDA; later PyTorch code will use Apple MPS when available and otherwise
fall back to CPU.

## 1. Create the environment

Install Python 3.11 (for example with `pyenv` or Homebrew), then run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
```

The complete dependency set includes PyTorch and Jupyter and may download
several hundred megabytes. Do not install it on a metered connection without
reviewing the package sizes first.

## 2. Download and verify EuroSAT RGB

The automated command downloads the authors' 94.7 MB RGB-only archive from
[Zenodo record 7711810](https://zenodo.org/records/7711810), verifies its
published MD5 checksum, extracts it safely, and rejects the 13-band dataset:

```bash
python -m src.data download
python -m src.data prepare
```

`prepare` decodes every image, verifies the 10 exact class counts and 64x64 RGB
shape, and creates deterministic 70/15/15 manifests under
`data/processed/splits/` using seed 42. Existing manifests are not overwritten
unless `--force` is explicitly supplied.

For a faster integrity check that still verifies every class count but decodes
only 100 deterministically sampled files:

```bash
python -m src.data prepare --sample-images 100
```

For a network failure, download `EuroSAT_RGB.zip` manually from the Zenodo page,
place it in `data/raw/`, and run:

```bash
python -m src.data download --archive data/raw/EuroSAT_RGB.zip
python -m src.data prepare
```

The command will report a checksum mismatch rather than accepting the wrong or
incomplete archive.

## 3. Generate exploratory figures

After the manifests exist:

```bash
python -m src.preprocessing
jupyter lab notebooks/01_eda.ipynb
```

The command saves `results/figures/class_distribution.png` and
`results/figures/example_grid.png`. The notebook is intentionally lightweight:
it calls the tested source modules rather than duplicating data logic.

## 4. Modes and experiment boundary

`configs/default.yaml` records the shared seed, source checksum, expected class
counts, split fractions, and a 100-images-per-class smoke setting. Phase 3–5
training commands will consume these same manifests. Model selection may use
only `train.csv` and `validation.csv`; `test.csv` remains untouched until the
chosen configurations receive their final evaluation.

## 5. Phase 3: traditional machine learning

Audit the full Phase 2 validation record and the exact, disjoint split
partition before training:

```bash
python -m src.train_classical audit
```

Run all four pipelines on a balanced smoke subset (100 training and 50
validation images per class):

```bash
python -m src.train_classical train --mode smoke --run-name smoke_phase3
```

This trains Random Forest, Random Forest + PCA, RBF SVM, and RBF SVM + PCA.
Scalers and PCA are inside scikit-learn pipelines and are fitted only by
`pipeline.fit(X_train, y_train)`. The command never loads test image features.
It saves configuration/runtime metadata, every validation candidate, macro and
per-class metrics, timings, selected train-only estimators, confusion matrices,
and a comparison chart under the configured `results/` directories.

Full Random Forest experiments use all 18,900 training and 4,050 validation
images:

```bash
python -m src.train_classical train --mode full \
  --models random_forest random_forest_pca \
  --run-name full_rf_phase3
```

The reviewed full SVM experiment uses PCA before the RBF kernel. A raw-pixel
full RBF SVM is intentionally not launched on an 8 GB Mac because of its
quadratic kernel cost and large support-vector footprint:

```bash
python -m src.train_classical train --mode full \
  --models svm_pca \
  --run-name full_svm_pca_phase3_20261115 \
  --allow-expensive-full-svm
```

For a non-PCA baseline under the same resource limit, use a separately named
full-data linear SVM or a predeclared stratified raw-RBF subset; do not describe
either as the full raw-RBF experiment.

Feature matrices are cached under ignored `data/processed/features/` by
default. Add `--no-feature-cache` when disk space is more constrained than
decode time. Each run name must be unique so earlier measurements are not
silently overwritten.

## 6. Phase 4: PyTorch CNN

Verify the full Phase 2 partition without creating image tensors:

```bash
python -m src.train_cnn audit
```

Run the bounded smoke configuration (500 training images, 200 validation
images, two epochs):

```bash
python -m src.train_cnn train --mode smoke --run-name smoke_cnn_phase4
```

The CNN uses three convolution/ReLU/max-pooling stages, adaptive pooling, and a
two-layer classification head. RGB normalization statistics are calculated
from the selected training identities only. PyTorch uses Apple MPS when it is
available and otherwise falls back to CPU. The command saves every epoch's
loss, accuracy, macro F1 and duration, a best-validation checkpoint, learning
curves, a validation confusion matrix, and full runtime/configuration metadata.
It never constructs a test dataset.

Full mode uses all 18,900 training and 4,050 validation images, batch size 64,
up to 30 epochs, and early stopping patience 5. It is blocked unless resources
have been checked and the explicit acknowledgement is supplied:

```bash
python -m src.train_cnn train \
  --mode full \
  --run-name full_cnn_phase4 \
  --allow-full-training
```

Use a unique run name. Full training is not an implicit authorization to load
or evaluate the held-out test images.

## 7. Phase 5: frozen final evaluation

Verify artifact hashes, fitted preprocessing, selected candidates/checkpoint,
seeds, class order, and validation-only provenance without reading the test
manifest or opening test images:

```bash
python -m src.evaluate inventory
```

The frozen inventory lives in `configs/final_evaluation.yaml`. The separate
final command is intentionally guarded and must be run only after explicit
approval:

```bash
python -m src.evaluate final-test \
  --run-name final_test_phase5 \
  --confirm-final-test
```

The command loads the existing train-only selected artifacts and never calls a
training method. It scores all four models on the same 4,050 identities, writes
accuracy, macro metrics, per-class metrics, numeric and PNG confusion matrices,
inference timings, comparison artifacts, runtime metadata and provenance, then
records a global completion state. That state prevents a silent second test
evaluation. A failed evaluation also remains locked pending explicit review.

## 8. Reproducing the published artifacts from a clean clone

The repository intentionally excludes EuroSAT images, feature caches, Conda or
virtual environments, and serialized `.joblib`/`.pt` models. It includes all
source code, exact configurations, selected validation metrics, final test
metrics, figures, artifact hashes, and runtime metadata.

For the closest software match, use Python 3.11.17 and install the tested lock:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pytest
```

Then download/prepare EuroSAT as described above and regenerate the selected
train-only artifacts using the paths frozen by `configs/final_evaluation.yaml`:

```bash
python -m src.train_classical train --mode full \
  --models random_forest random_forest_pca \
  --run-name full_rf_phase3_20261115

python -m src.train_classical train --mode full \
  --models svm_pca \
  --run-name full_svm_pca_phase3_20261115 \
  --allow-expensive-full-svm

python -m src.train_cnn train --mode full \
  --run-name full_cnn_phase4_20261115_v2 \
  --allow-full-training
```

The committed final metrics are the authoritative one-shot test result. The
published checkpoint hashes identify the original local artifacts but those
large binaries are not distributed. Byte-for-byte hashes of regenerated models
can differ across platform or library builds even when predictions match. To
perform an independent reproduction, copy `configs/final_evaluation.yaml` to a
new filename, update only the copied artifact hashes after verifying the frozen
validation metrics, and assign a different state/output path. Preserve the
committed configuration and `final_test_state.json` as publication provenance;
never delete them merely to bypass the one-shot guard.
