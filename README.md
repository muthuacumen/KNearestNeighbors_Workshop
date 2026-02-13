# K-Nearest Neighbors Workshop — Solution

**Course:** Machine Learning with Python
**Instructor:** David Espinosa
**Release:** October 2025
**Solution completed:** February 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Workshop Completion Status](#workshop-completion-status)
3. [Repository Structure](#repository-structure)
4. [Dataset Source](#dataset-source)
5. [Environment Setup](#environment-setup)
6. [Example Usage](#example-usage)
7. [Pipeline Architecture](#pipeline-architecture)
8. [Preprocessing Steps](#preprocessing-steps)
9. [Model Training and Evaluation](#model-training-and-evaluation)
10. [Hyperparameter Tuning](#hyperparameter-tuning)
11. [Testing and Debugging](#testing-and-debugging)
12. [Function Reference](#function-reference)
13. [Browser UI — Interactive Risk Predictor](#browser-ui--interactive-risk-predictor)
14. [Results](#results)
15. [Reflections and Peer Observations](#reflections-and-peer-observations)
16. [KNN Algorithm Overview](#knn-algorithm-overview)

---

## Workshop Completion Status

| Part | Title | Status |
|------|-------|--------|
| Part 1 | Warm-up with the Iris Dataset | ✅ Solved |
| **Part 2** | **Apply KNN on a Real-World Dataset** | ✅ **Solved** |
| Part 3 | Architecting with the ML Pipeline Pattern | ✅ Solved (see `KNN_Workshop_Solution.ipynb`) |

### Part 2 — Solution Summary

**Part 2** (`KNearestNeighbors_Workshop.ipynb`, cell `a43689b2`) is fully implemented.

**Dataset:** Open-Meteo 14-day hourly weather forecast for Berlin (no API key required)
**Source:** `https://api.open-meteo.com/v1/forecast`
**Samples:** 336 hourly observations (14 days × 24 hours)

**Engineered target — `ThermalComfort` (3 classes):**

| Class | Condition |
|---|---|
| **Cold** | temperature < 10 °C |
| **Comfortable** | 10 °C ≤ temperature ≤ 25 °C |
| **Hot** | temperature > 25 °C |

**Features used:**

| Feature | Description |
|---|---|
| `temperature_2m` | Air temperature at 2 m height (°C) |
| `relative_humidity_2m` | Relative humidity at 2 m (%) |
| `wind_speed_10m` | Wind speed at 10 m height (km/h) |
| `apparent_temperature` | Feels-like temperature (°C) |
| `precipitation_probability` | Precipitation probability (%) |
| `hour` | Hour of day (0–23) |
| `day_of_week` | Day of week (0 = Monday) |

**Pipeline steps implemented in the cell:**

1. **Step 1 — Acquire**: `requests.get()` to the Open-Meteo API, parsed into a `pd.DataFrame`
2. **Step 2 — Feature engineering**: time-based features (`hour`, `day_of_week`) + `ThermalComfort` label
3. **Step 3 — Preprocess**: median imputation for any missing values, `StandardScaler`, `LabelEncoder`, stratified 80/20 train-test split
4. **Step 4 — Baseline KNN**: `Pipeline(StandardScaler + KNeighborsClassifier(k=5, Euclidean))`, classification report + confusion matrix
5. **Step 5 — Hyperparameter tuning**: `GridSearchCV` with `StratifiedKFold(n_splits=5)` over 16 parameter combinations (k ∈ {3,5,7,9} × metric ∈ {euclidean, manhattan} × weights ∈ {uniform, distance}), scored by `f1_weighted`; outputs comparison bar chart

---

## Overview

This project implements a complete, production-quality **K-Nearest Neighbors (KNN) classification pipeline** for a medical risk pre-screening task.

Given a patient's medical measurements — BMI, blood pressure, glucose level, age, and others — the model predicts whether their health risk is **Low**, **Medium**, or **High**.

The implementation follows the **ML Pipeline Pattern**: each stage (data acquisition, preprocessing, training, evaluation, tuning, testing) is encapsulated in a self-contained, independently callable function. This mirrors professional MLOps practice and makes the code modular, testable, and reproducible.

**Key design decisions:**

| Decision | Rationale |
|---|---|
| `sklearn.pipeline.Pipeline` wrapping `StandardScaler + KNN` | Prevents data leakage: scaler fits only on training data per fold |
| `StratifiedKFold` in GridSearchCV | Minority `Low` class (~6%) must appear in every CV fold |
| `f1_weighted` as tuning metric | More honest than accuracy for imbalanced 3-class data |
| `random_state=42` throughout | Full reproducibility across runs |
| Assert-based sanity checks | Fail loudly with actionable messages rather than silently producing wrong results |

---

## Repository Structure

```
KNearestNeighbors_Workshop/
│
├── KNearestNeighbors_Workshop.ipynb   # Original workshop notebook (do not modify)
├── KNN_Workshop_Solution.ipynb        # Solution notebook — all 6 pipeline stages
├── knn_pipeline.py                    # Standalone VS Code-ready Python script
├── medical_risk_predictor.html        # Browser UI (auto-generated when Part 5A cell runs)
├── README.md                          # This file
└── requirements.txt                   # Pinned dependencies
```

---

## Dataset Source

**Dataset:** Pima Indians Diabetes
**Samples:** 768
**Features:** 8 numerical medical measurements
**Origin:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
**Access:** Public CSV hosted at `github.com/jbrownlee/Datasets` (MIT licence, UCI ML Repository mirror)
**Fetched via:** `requests.get()` — no authentication required

### Features

| Column | Description | Unit |
|---|---|---|
| `Pregnancies` | Number of pregnancies | count |
| `Glucose` | Plasma glucose concentration (2h OGTT) | mg/dL |
| `BloodPressure` | Diastolic blood pressure | mm Hg |
| `SkinThickness` | Triceps skinfold thickness | mm |
| `Insulin` | 2-hour serum insulin | mu U/mL |
| `BMI` | Body mass index | kg/m² |
| `DiabetesPedigreeFunction` | Diabetes hereditary score | — |
| `Age` | Age | years |

### Engineered Target: Risk Label

The original dataset has a binary diabetes outcome. We engineer a clinically-meaningful **3-class risk label** from three key features using WHO/CDC-aligned thresholds:

| Risk | Condition |
|---|---|
| **High** | Glucose > 140 **or** BMI > 35 **or** BloodPressure > 100 |
| **Low** | Glucose < 100 **and** BMI < 25 **and** BloodPressure < 80 |
| **Medium** | All other cases |

Resulting distribution: **High ≈ 356**, **Medium ≈ 369**, **Low ≈ 43**
The `Low` class is a minority (~6%) — handled with stratified splits and weighted F1 scoring.

---

## Environment Setup

**Requirements:** Python 3.9+

```bash
# Clone the repository
git clone https://github.com/ProfEspinosaAIML/KNearestNeighbors_Workshop.git
cd KNearestNeighbors_Workshop

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Core dependencies

| Package | Version | Purpose |
|---|---|---|
| `scikit-learn` | ≥ 1.3 | KNN, Pipeline, GridSearchCV, metrics |
| `pandas` | ≥ 2.0 | DataFrame manipulation |
| `numpy` | ≥ 1.24 | Numerical operations |
| `matplotlib` | ≥ 3.7 | Visualisations |
| `requests` | ≥ 2.28 | HTTP dataset fetch |

---

## Example Usage

### Run the full pipeline from the command line

```bash
python knn_pipeline.py
```

### Use individual functions in a notebook or script

```python
from knn_pipeline import (
    load_data,
    preprocess_data,
    train_knn,
    evaluate_model,
    tune_knn,
    run_sanity_checks,
    main,
)

# ── 1. Acquire data ──────────────────────────────────────────────────────────
df_raw = load_data()
# Output: 768 rows × 10 columns including engineered 'Risk' column

# ── 2. Preprocess ────────────────────────────────────────────────────────────
X_scaled, y_encoded, scaler, label_encoder = preprocess_data(df_raw)
# Output: X_scaled (768, 8) float64, y_encoded (768,) int — High=0,Low=1,Medium=2

# ── 3. Split ─────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

# ── 4. Train (baseline k=5) ──────────────────────────────────────────────────
model = train_knn(X_train, y_train, n_neighbors=5, metric="minkowski")

# ── 5. Evaluate ──────────────────────────────────────────────────────────────
metrics = evaluate_model(model, X_test, y_test, label_encoder)
# Prints: Accuracy, Precision, Recall, F1 + Confusion Matrix

# ── 6. Tune hyperparameters ──────────────────────────────────────────────────
best_model, best_params, cv_results = tune_knn(X_train, y_train, cv=5)
print(best_params)
# e.g. {'knn__metric': 'manhattan', 'knn__n_neighbors': 7, 'knn__weights': 'distance'}

# ── 7. Evaluate tuned model ──────────────────────────────────────────────────
tuned_metrics = evaluate_model(best_model, X_test, y_test, label_encoder)

# ── 8. Decode a prediction ───────────────────────────────────────────────────
import numpy as np
sample = X_test[:1]                                   # shape (1, 8)
pred_int   = best_model.predict(sample)               # e.g. [2]
pred_label = label_encoder.inverse_transform(pred_int) # e.g. ['Medium']
print(f"Risk prediction: {pred_label[0]}")

# ── 9. Run full pipeline in one call ─────────────────────────────────────────
results = main()
# Returns dict with model, X_train, X_test, y_train, y_test,
#                       scaler, label_encoder, metrics, df_raw, X, y
```

---

## Pipeline Architecture

```
load_data()
    │  HTTP GET → public CSV → pd.DataFrame (768 × 10)
    │  Engineer Risk label from clinical thresholds
    ▼
preprocess_data(df)
    │  1. Replace impossible zeros with NaN
    │  2. Median imputation (numerical)
    │  3. Mode  imputation (categorical)
    │  4. StandardScaler  → X_scaled (768 × 8)
    │  5. LabelEncoder    → y_encoded (768,)
    ▼
train_test_split(X_scaled, y_encoded, test_size=0.20,
                 random_state=42, stratify=y_encoded)
    │  614 train  /  154 test
    ▼
train_knn(X_train, y_train)                tune_knn(X_train, y_train)
    │  Pipeline:                               │  GridSearchCV:
    │  StandardScaler (fit on X_train only)    │  StratifiedKFold(n=5)
    │  KNeighborsClassifier(k=5)               │  16 param combos × 5 folds
    │                                          │  = 80 CV fits
    ▼                                          ▼
evaluate_model(model, X_test, y_test)     evaluate_model(best_model, ...)
    │  Accuracy, Precision, Recall, F1         │  Compare vs baseline
    │  Classification report                   │
    │  Confusion matrix plot                   │
    ▼                                          ▼
run_sanity_checks(...)
    │  16 assert-based checks
    │  Shape, NaN, labels, split, metrics
    ▼
main()  ←  orchestrates all stages end-to-end
```

---

## Preprocessing Steps

### 1. Replace Impossible Zeros

Several columns contain `0` as a data-entry placeholder for missing values.
A glucose level of 0 or a BMI of 0 is physiologically impossible.

```python
ZERO_AS_NAN_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
```

| Column | Zeros replaced |
|---|---|
| Glucose | 5 |
| BloodPressure | 35 |
| SkinThickness | 227 |
| Insulin | 374 |
| BMI | 11 |

### 2. Median Imputation

Median is used instead of mean because clinical features like Insulin have extreme outliers (max 846 µU/mL). The mean would be pulled upward, creating unrealistically high imputed values.

### 3. Mode Imputation

Applied to any categorical columns with missing values. The `Risk` column has no NaN (it is engineered from clean numeric data), but the check is included defensively.

### 4. StandardScaler

KNN classifies by **Euclidean distance**. Without scaling, a feature with a large range — like `Insulin` (0–846) — would dominate every distance calculation, effectively making the other 7 features irrelevant.

`StandardScaler` transforms each feature to **zero mean and unit variance**:

```
z = (x - mean) / std
```

The scaler is fitted **only on the training fold** inside the `Pipeline` to prevent leakage.

### 5. LabelEncoder

Converts the string target to integers (alphabetical order):

```
High → 0     Low → 1     Medium → 2
```

Use `label_encoder.inverse_transform(predictions)` to convert back to strings.

---

## Model Training and Evaluation

### Training

`train_knn()` wraps `StandardScaler + KNeighborsClassifier` in a `sklearn.pipeline.Pipeline`:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("knn",    KNeighborsClassifier(n_neighbors=5, metric="minkowski", n_jobs=-1)),
])
```

**Default configuration:**

| Parameter | Value | Meaning |
|---|---|---|
| `n_neighbors` | 5 | Vote taken from 5 nearest neighbours |
| `metric` | `minkowski` (p=2) | Euclidean distance |
| `weights` | `uniform` | All neighbours vote equally |
| `n_jobs` | -1 | Use all CPU cores for distance computation |

### Evaluation Metrics

All metrics use `average='weighted'` to account for class imbalance.

| Metric | Formula | Why used |
|---|---|---|
| **Accuracy** | correct / total | Overall correctness |
| **Precision** | TP / (TP + FP) | Fraction of positive predictions that are correct |
| **Recall** | TP / (TP + FN) | Fraction of actual positives correctly identified |
| **F1-score** | 2 × P × R / (P + R) | Harmonic mean of precision and recall |

Weighted averaging weights each class's metric by its support (number of samples), so the large `Medium`/`High` classes do not unfairly dominate.

### Output

- Printed classification report (per-class precision, recall, F1)
- Confusion matrix heatmap (matplotlib)

---

## Hyperparameter Tuning

`tune_knn()` performs an exhaustive grid search using `GridSearchCV` with 5-fold stratified cross-validation.

### Parameter Grid

| Parameter | Values | Total |
|---|---|---|
| `knn__n_neighbors` | 3, 5, 7, 9 | 4 |
| `knn__metric` | `euclidean`, `manhattan` | 2 |
| `knn__weights` | `uniform`, `distance` | 2 |
| **Combinations** | | **16** |
| **CV folds** | 5 | |
| **Total fits** | | **80** |

### Why These Values?

**`n_neighbors` [3, 5, 7, 9]** — Odd values to avoid ties. Small k risks overfitting (noise-sensitive); large k risks underfitting (over-smoothed boundaries).

**`euclidean` vs `manhattan`**
Euclidean (L2) penalises large deviations quadratically. For clinical data with extreme outliers (e.g. Insulin), Manhattan (L1) can be more robust because it treats every unit of deviation linearly.

**`uniform` vs `distance`**
`distance` weighting gives closer neighbours proportionally higher voting power, often improving accuracy near decision boundaries.

### Scoring Metric: `f1_weighted`

`accuracy` can be misleading when classes are imbalanced. With `Low` representing only ~6% of samples, a model that never predicts `Low` still achieves ~94% accuracy. `f1_weighted` penalises such behaviour by incorporating per-class precision and recall.

### Implementation Detail: Leakage Prevention

```python
# GridSearchCV wraps the full Pipeline, not just KNN.
# On every fold: scaler.fit(X_train_fold) → scaler.transform(X_val_fold)
# The validation fold NEVER influences the scaler's mean/std.
grid_search = GridSearchCV(estimator=Pipeline([("scaler", ...), ("knn", ...)]), ...)
```

---

## Testing and Debugging

`run_sanity_checks()` runs 16 assert-based checks that raise `AssertionError` with a descriptive `FIX →` hint on failure.

| # | Check | Silent failure prevented |
|---|---|---|
| 1 | `X` rows == `y` rows | Misaligned training index |
| 2 | No NaN in `X` | NaN distances → garbage predictions |
| 3 | Correct feature count | Wrong dimensionality fed to model |
| 4 | Dataset >= 500 samples | Truncated HTTP fetch accepted silently |
| 5 | Classes == {High, Low, Medium} | Extra label shifts all encoded integers |
| 6 | Encoded y in [0, n\_classes) | `inverse_transform` IndexError |
| 7 | Train + Test == Total | Dropped/duplicated rows bias evaluation |
| 8 | Test ratio ≈ 20% | Accidental TEST\_SIZE change |
| 9 | Both splits contain all classes | Undefined recall for absent class |
| 10 | Smoke-test fit succeeds | dtype / shape bug before 80-fit search |
| 11 | `len(y_pred)` == `len(X_test)` | Metric indexing error |
| 12 | All predictions valid integers | `inverse_transform` crash |
| 13–16 | All metrics in [0, 1] | Wrong `average=` argument |

---

## Function Reference

### `load_data() → pd.DataFrame`

Fetch the Pima Indians Diabetes dataset via HTTP GET and engineer a 3-class risk label.

| | |
|---|---|
| **Inputs** | None |
| **Returns** | `pd.DataFrame` — 768 rows × 10 columns |
| **Side effects** | Network request; prints shape and class distribution |
| **Raises** | `requests.HTTPError` on non-2xx response |

---

### `preprocess_data(df) → (X_scaled, y_encoded, scaler, label_encoder)`

Clean, impute, scale, and encode the raw DataFrame.

| | |
|---|---|
| **Input** | `df: pd.DataFrame` — raw output of `load_data()` |
| **Returns** | `X_scaled` (np.ndarray, 768×8), `y_encoded` (np.ndarray, 768), `StandardScaler`, `LabelEncoder` |
| **Deterministic** | Yes — no randomness |

---

### `train_knn(X_train, y_train, n_neighbors=5, metric='minkowski', weights='uniform') → Pipeline`

Build and fit a `StandardScaler + KNeighborsClassifier` sklearn Pipeline.

| | |
|---|---|
| **Inputs** | `X_train` (np.ndarray), `y_train` (np.ndarray), `n_neighbors` (int), `metric` (str), `weights` (str) |
| **Returns** | Fitted `sklearn.pipeline.Pipeline` |
| **Note** | Scaler is fit on `X_train` only — no leakage |

---

### `evaluate_model(model, X_test, y_test, label_encoder=None) → dict`

Evaluate a fitted Pipeline on the test set.

| | |
|---|---|
| **Inputs** | `model` (Pipeline), `X_test` (np.ndarray), `y_test` (np.ndarray), `label_encoder` (LabelEncoder, optional) |
| **Returns** | `dict` — `{'accuracy', 'precision', 'recall', 'f1'}` |
| **Side effects** | Prints classification report; shows confusion matrix plot |

---

### `tune_knn(X_train, y_train, cv=5, scoring='f1_weighted', verbose=False) → (best_model, best_params, cv_results)`

Exhaustive grid search over k, metric, and weights with stratified CV.

| | |
|---|---|
| **Inputs** | `X_train`, `y_train`, `cv` (int), `scoring` (str), `verbose` (bool) |
| **Returns** | `(Pipeline, dict, pd.DataFrame)` — best model, best params, full results table |
| **CV fits** | 80 (16 combos × 5 folds) |
| **Leakage-safe** | Yes — scaler inside Pipeline, refitted per fold |

---

### `run_sanity_checks(df_raw, X, y, X_train, X_test, y_train, y_test, model, label_encoder) → dict`

Assert-based validation across all pipeline artefacts.

| | |
|---|---|
| **Inputs** | All major pipeline artefacts |
| **Returns** | `dict` — `{'accuracy', 'precision', 'recall', 'f1'}` |
| **Raises** | `AssertionError` with `FIX →` hint on first failure |

---

### `main() → dict`

Orchestrate all pipeline stages end-to-end (Stages 1–5).

| | |
|---|---|
| **Inputs** | None |
| **Returns** | `dict` with keys: `model`, `df_raw`, `X`, `y`, `X_train`, `X_test`, `y_train`, `y_test`, `scaler`, `label_encoder`, `metrics` |

---

## Browser UI — Interactive Risk Predictor

**Part 5A** of the notebook adds a fully self-contained browser interface for real-time risk prediction.

### How It Works

```
Notebook cell (Part 5A)
    │
    ├── start_prediction_server(model, scaler, label_encoder, port=8765)
    │       Spawns a Python HTTP server in a daemon thread.
    │       Accepts POST /predict with JSON body → returns JSON prediction.
    │
    ├── build_html(port=8765)
    │       Generates a single-file HTML page with inline CSS and JS.
    │       Writes it to medical_risk_predictor.html.
    │
    └── webbrowser.open()  +  IPython.display.HTML inline link
            Opens the predictor in the default browser.
```

### Prediction Server

| Detail | Value |
|---|---|
| Port | `8765` (configurable) |
| Endpoint | `POST /predict` |
| Request body | JSON with feature keys (see below) |
| Response body | `{"risk": "High"\|"Medium"\|"Low", "confidence": float, "class_probs": {...}}` |
| Thread | Daemon thread — stops automatically when the notebook kernel shuts down |
| Re-run safety | Global `_prediction_server` handle; existing server is shut down before starting a new one |
| CORS | `Access-Control-Allow-Origin: *` — required for `file://` → `localhost` fetch |

**Example request:**
```json
{
  "BMI": 32.5,
  "Age": 45,
  "BloodPressure": 88,
  "Glucose": 120,
  "Insulin": 0,
  "SkinThickness": 0,
  "Pregnancies": 2,
  "DiabetesPedigreeFunction": 0.5
}
```

**Example response:**
```json
{
  "risk": "Medium",
  "confidence": 0.714,
  "class_probs": {"High": 0.143, "Low": 0.0, "Medium": 0.714, "Unknown": 0.143}
}
```

Fields `Glucose`, `Insulin`, `SkinThickness`, `Pregnancies`, and `DiabetesPedigreeFunction` are **optional** — the server fills missing values with the training-set median before scaling.

### UI Features

| Element | Description |
|---|---|
| Required fields | BMI, Age, Blood Pressure — highlighted in blue, validated before submit |
| Optional fields | Glucose, Insulin, Skin Thickness, Pregnancies, Diabetes Pedigree Function |
| Result box | Colour-coded: green = Low risk, orange = Medium, red = High |
| Confidence bar | Colour-matched progress bar showing prediction confidence % |
| Probability chips | Per-class percentage badges (Low / Medium / High) |
| Loading state | Spinner shown during fetch |
| Error handling | Toast message if the prediction server is unreachable |
| Medical disclaimer | Displayed below result; tool is for educational use only |

### Function Reference

#### `start_prediction_server(model, scaler, label_encoder, port=8765) → HTTPServer`

| | |
|---|---|
| **Inputs** | Fitted `Pipeline`, `StandardScaler`, `LabelEncoder`, port number |
| **Returns** | `http.server.HTTPServer` instance |
| **Side effects** | Starts daemon thread; shuts down any previously running server on the same port |

#### `build_html(port=8765) → str`

| | |
|---|---|
| **Inputs** | `port` (int) |
| **Returns** | Complete HTML string (no external CDN dependencies) |
| **Side effects** | Writes `medical_risk_predictor.html` to the working directory |

### Running the UI Without the Notebook

Start the prediction server from `knn_pipeline.py`, then open the HTML file in any browser:

```python
from knn_pipeline import main
from your_notebook_utils import start_prediction_server, build_html

results = main()
start_prediction_server(results["model"], results["scaler"], results["label_encoder"])
build_html()
# Open medical_risk_predictor.html in your browser
```

---

## Results

> Results below are representative; exact values depend on the tuned hyperparameters found during GridSearchCV.

### Baseline (k=5, Euclidean, uniform weights)

| Metric | Score |
|---|---|
| Accuracy | ~0.84 |
| Precision (weighted) | ~0.83 |
| Recall (weighted) | ~0.84 |
| F1-weighted | ~0.83 |

### After Tuning (GridSearchCV, f1_weighted, cv=5)

| Metric | Score |
|---|---|
| Accuracy | ~0.86–0.89 |
| Precision (weighted) | ~0.86–0.88 |
| Recall (weighted) | ~0.86–0.89 |
| F1-weighted | ~0.86–0.88 |

> Typical best params: `{'knn__metric': 'manhattan', 'knn__n_neighbors': 7, 'knn__weights': 'distance'}`

---

## Reflections and Peer Observations

### KNN Performance on This Dataset

KNN performs well on the Pima dataset because the features are all numerical and the risk boundaries can be expressed approximately as Voronoi regions in 8-dimensional space. After tuning, weighted F1 scores in the 0.86–0.89 range suggest good generalisation.

### Preprocessing Impact

The single highest-impact preprocessing step was **replacing impossible zeros with NaN before scaling**. Without this, zero-insulin patients pulled the Insulin mean down significantly, degrading the scaler's normalisation. Median imputation further stabilised this.

### Strengths of KNN for This Task

- No assumptions about feature distribution (non-parametric)
- Simple to interpret: "this patient's risk matches the k most similar patients in the training set"
- Naturally handles multi-class classification

### Weaknesses of KNN

- **Prediction is slow at inference time**: O(n) distance computations for each new patient
- **Memory-intensive**: must store all training samples
- **Sensitive to irrelevant features**: features like `DiabetesPedigreeFunction` may add noise
- **Class imbalance**: the `Low` class (43 samples) is consistently harder to classify

### When Not to Use KNN

- Large datasets (>100k rows) — inference becomes prohibitively slow
- High-dimensional sparse data — distance metrics lose meaning (curse of dimensionality)
- Real-time latency requirements — tree-based or linear models are faster at inference

### MLOps Extension

In production, this pipeline could be extended with:
- `joblib.dump(best_model, 'knn_model.pkl')` for model persistence
- FastAPI endpoint wrapping `best_model.predict()`
- Automated retraining triggered when data drift is detected (e.g. Evidently AI)
- CI/CD: `run_sanity_checks()` as a pre-deployment gate in a GitHub Actions workflow

---

## KNN Algorithm Overview

K-Nearest Neighbors is a **lazy, non-parametric, instance-based** learning algorithm.

**At training time:** the algorithm simply stores all labelled training samples.

**At prediction time for a new point `q`:**
1. Compute the distance from `q` to every training point
2. Select the `k` closest points (nearest neighbours)
3. Assign the **majority class** among those `k` neighbours

**Distance metrics:**

| Metric | Formula | Notes |
|---|---|---|
| Euclidean (L2) | `sqrt(sum((xi - qi)²))` | Default; penalises large gaps quadratically |
| Manhattan (L1) | `sum(|xi - qi|)` | More robust to outliers in clinical data |

**Choosing k:**

```
k too small → overfits to noise in training data
k too large → under-smoothed boundaries, ignores local structure
k = sqrt(n) → common heuristic starting point (√614 ≈ 25 for this dataset)
```

**Interactive simulator:** [k-nearest-neighbors.vercel.app](https://k-nearest-neighbors.vercel.app/)

---

*Workshop materials by David Espinosa — AI/ML Engineering Programme*
