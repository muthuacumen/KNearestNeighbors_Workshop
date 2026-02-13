"""
knn_pipeline.py
===============
Medical Risk Pre-Screening — KNN Pipeline
==========================================

VS Code-ready standalone script implementing the full ML pipeline pattern
for KNN-based medical risk classification (Low / Medium / High).

Dataset
-------
Pima Indians Diabetes dataset — 768 samples, 8 numerical medical features
fetched from a public CSV endpoint via `requests`.

Pipeline Stages
---------------
1. load_data()          → acquire raw DataFrame from public API
2. preprocess_data(df)  → clean, impute, scale, encode
3. train_knn(...)       → build + fit sklearn Pipeline (Scaler + KNN)
4. evaluate_model(...)  → metrics report + confusion matrix
5. main()               → orchestrate all stages end-to-end

Usage
-----
    python knn_pipeline.py

Requirements
------------
    pip install scikit-learn pandas numpy matplotlib requests
"""

from __future__ import annotations

import sys
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

# ── Global constants ─────────────────────────────────────────────────────────
DATASET_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

COLUMN_NAMES: list[str] = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]

# Columns where a value of 0 is physiologically impossible (treated as missing)
ZERO_AS_NAN_COLS: list[str] = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]

# Feature columns fed to the model
FEATURE_COLS: list[str] = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

TARGET_COL:   str = "Risk"
RANDOM_STATE: int = 42        # seed for reproducibility
TEST_SIZE:    float = 0.20    # 80 / 20 train-test split


# ════════════════════════════════════════════════════════════════════════════
# Stage 1 — Data Acquisition
# ════════════════════════════════════════════════════════════════════════════

def _assign_risk_label(row: pd.Series) -> str:
    """
    Assign a 3-class risk label based on WHO/CDC-aligned clinical thresholds.

    Rules (applied in priority order)
    ----------------------------------
    High   : Glucose > 140  OR  BMI > 35  OR  BloodPressure > 100
    Low    : Glucose < 100  AND BMI < 25  AND BloodPressure < 80
    Medium : all other cases

    Parameters
    ----------
    row : pd.Series
        A single patient record with at minimum Glucose, BMI, BloodPressure.

    Returns
    -------
    str : one of 'High', 'Medium', 'Low'
    """
    if row["Glucose"] > 140 or row["BMI"] > 35 or row["BloodPressure"] > 100:
        return "High"
    if row["Glucose"] < 100 and row["BMI"] < 25 and row["BloodPressure"] < 80:
        return "Low"
    return "Medium"


def load_data() -> pd.DataFrame:
    """
    Acquire the Pima Indians Diabetes dataset from a public CSV endpoint
    using the `requests` library.

    Steps
    -----
    - Fetch raw CSV via HTTP GET (no authentication required)
    - Assign human-readable column names
    - Engineer a 3-class Risk label (Low / Medium / High) from clinical rules

    Returns
    -------
    pd.DataFrame
        768 rows × 10 columns.
        Columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                 BMI, DiabetesPedigreeFunction, Age, Outcome, Risk

    Raises
    ------
    requests.HTTPError
        If the remote endpoint returns a non-2xx status code.
    """
    print(f"[load_data] Fetching dataset from:\n  {DATASET_URL}\n")
    response = requests.get(DATASET_URL, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text), header=None, names=COLUMN_NAMES)

    # Engineer 3-class risk label from clinical thresholds
    df[TARGET_COL] = df.apply(_assign_risk_label, axis=1)

    print(f"[load_data] Loaded {len(df)} samples, {df.shape[1]} columns.")
    print(f"  Risk distribution:\n{df[TARGET_COL].value_counts().to_string()}\n")
    print("First 5 rows:")
    print(df[["Age", "BMI", "BloodPressure", "Glucose", "Insulin", TARGET_COL]].head().to_string())
    print()
    return df


# ════════════════════════════════════════════════════════════════════════════
# Stage 2 — Preprocessing
# ════════════════════════════════════════════════════════════════════════════

def preprocess_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
    """
    Clean, impute, scale, and encode the raw medical DataFrame.

    Steps
    -----
    1. Replace physiologically impossible zeros with NaN.
       (Zero is not a valid reading for Glucose, BMI, BloodPressure, etc.)
    2. Median imputation for numerical NaN — robust to clinical outliers.
    3. Mode imputation for any categorical NaN — preserves dominant class.
    4. StandardScaler: transform features to zero mean / unit variance.
       Critical for KNN because it relies on Euclidean distance; without
       scaling, high-range features (e.g. Insulin 0–846) dominate the metric.
    5. LabelEncoder: map 'High'/'Low'/'Medium' → integers (alphabetical order).

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame returned by load_data().

    Returns
    -------
    X_scaled : np.ndarray, shape (768, 8)
        Scaled feature matrix.
    y_encoded : np.ndarray, shape (768,)
        Integer-encoded target: High=0, Low=1, Medium=2
    scaler : StandardScaler
        Fitted scaler (retain for future inference on unseen data).
    label_encoder : LabelEncoder
        Fitted encoder — use .inverse_transform() to decode predictions.

    Notes
    -----
    random_state=42 applies to train_test_split (Stage 3), not here.
    This function is fully deterministic.
    """
    df_clean = df.copy()

    # ── Step 1: Replace impossible zeros ────────────────────────────────
    print("[preprocess_data] Step 1 — replacing impossible zeros...")
    for col in ZERO_AS_NAN_COLS:
        n = (df_clean[col] == 0).sum()
        df_clean[col] = df_clean[col].replace(0, np.nan)
        print(f"  {col:<30s}  {n:>3d} zeros → NaN")

    # ── Step 2: Median imputation (numerical) ───────────────────────────
    print("[preprocess_data] Step 2 — median imputation (numerical)...")
    for col in FEATURE_COLS:
        n_nan = df_clean[col].isna().sum()
        if n_nan > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"  {col:<30s}  {n_nan:>3d} NaN → median={median_val:.4f}")

    # ── Step 3: Mode imputation (categorical) ───────────────────────────
    print("[preprocess_data] Step 3 — mode imputation (categorical)...")
    for col in df_clean.select_dtypes(include=["object", "category"]).columns:
        n_nan = df_clean[col].isna().sum()
        if n_nan > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            print(f"  {col:<30s}  {n_nan:>3d} NaN → mode='{mode_val}'")
        # (no NaN → silently skip)

    # ── Step 4: StandardScaler ──────────────────────────────────────────
    # NOTE: This scaler is fit on the full dataset for EDA/reference.
    # The Pipeline in train_knn() fits its own scaler on X_train only,
    # which is the correct approach for avoiding data leakage.
    print("[preprocess_data] Step 4 — StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[FEATURE_COLS].values)
    print(f"  X_scaled shape : {X_scaled.shape}")
    print(f"  NaN remaining  : {np.isnan(X_scaled).sum()}")

    # ── Step 5: LabelEncoder ────────────────────────────────────────────
    print("[preprocess_data] Step 5 — LabelEncoder on target...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df_clean[TARGET_COL])
    encoding = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(f"  Encoding : {encoding}")

    print(f"\n[preprocess_data] Complete — X: {X_scaled.shape}, y: {y_encoded.shape}\n")
    return X_scaled, y_encoded, scaler, label_encoder


# ════════════════════════════════════════════════════════════════════════════
# Stage 3 — Train
# ════════════════════════════════════════════════════════════════════════════

def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 5,
    metric: str = "minkowski",
    weights: str = "uniform",
) -> Pipeline:
    """
    Build and fit a sklearn Pipeline: StandardScaler → KNeighborsClassifier.

    Wrapping KNN in a Pipeline ensures StandardScaler is fit **only on
    X_train**, preventing data leakage that would inflate test set scores.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix (raw/imputed — Pipeline handles scaling).
    y_train : np.ndarray
        Integer-encoded training labels from preprocess_data().
    n_neighbors : int, default=5
        k — number of nearest neighbours to vote on each prediction.
        Use odd values to avoid tie-breaking. Tuned in Part 4.
    metric : str, default='minkowski'
        Distance metric for neighbour search.
        'minkowski' with p=2 (default) == Euclidean distance.
        Alternatives: 'manhattan' (p=1), 'chebyshev', 'cosine'.
    weights : str, default='uniform'
        'uniform'  → all k neighbours have equal vote weight.
        'distance' → nearer neighbours contribute proportionally more.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline. Steps: [('scaler', StandardScaler),
                                  ('knn',   KNeighborsClassifier)]
    """
    pipeline = Pipeline([
        # Scaler: fit on X_train, applied (transform only) to X_test
        ("scaler", StandardScaler()),

        # KNN classifier with parallelised distance computations
        ("knn", KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric,
            weights=weights,
            n_jobs=-1,      # use all available CPU cores
        )),
    ])

    pipeline.fit(X_train, y_train)

    knn = pipeline.named_steps["knn"]
    print(f"[train_knn] Pipeline fitted.")
    print(f"  k (n_neighbors) : {knn.n_neighbors}")
    print(f"  metric          : {knn.metric}")
    print(f"  weights         : {knn.weights}")
    print(f"  Training samples: {X_train.shape[0]}\n")
    return pipeline


# ════════════════════════════════════════════════════════════════════════════
# Stage 4 — Evaluate
# ════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder | None = None,
) -> dict:
    """
    Evaluate a trained Pipeline on the test set.

    Computes accuracy, weighted precision, recall, and F1-score.
    Prints a full classification report and plots a confusion matrix.

    Weighted averaging accounts for class imbalance
    (High=356, Medium=369, Low=43 in this dataset).

    Parameters
    ----------
    model : Pipeline
        Fitted pipeline from train_knn().
    X_test : np.ndarray
        Test feature matrix (raw/imputed — Pipeline handles scaling).
    y_test : np.ndarray
        True integer-encoded test labels.
    label_encoder : LabelEncoder, optional
        When provided, class names (High/Low/Medium) appear in the report
        and confusion matrix instead of raw integers.

    Returns
    -------
    metrics : dict
        {
            'accuracy' : float,
            'precision': float,  # weighted
            'recall'   : float,  # weighted
            'f1'       : float,  # weighted
        }
    """
    y_pred = model.predict(X_test)

    class_names = list(label_encoder.classes_) if label_encoder is not None else None

    # ── Compute metrics ──────────────────────────────────────────────────
    metrics = {
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall"   : recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1"       : f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}  (weighted)")
    print(f"  Recall    : {metrics['recall']:.4f}  (weighted)")
    print(f"  F1-Score  : {metrics['f1']:.4f}  (weighted)")
    print()
    print("Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=class_names,
        zero_division=0,
    ))

    # ── Confusion matrix ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=class_names,
        colorbar=True,
        ax=ax,
        cmap="Blues",
    )
    k = model.named_steps["knn"].n_neighbors
    ax.set_title(f"Confusion Matrix — KNN (k={k})", fontsize=12)
    plt.tight_layout()
    plt.show()

    return metrics


# ════════════════════════════════════════════════════════════════════════════
# Orchestrator — main()
# ════════════════════════════════════════════════════════════════════════════

def main() -> dict:
    """
    End-to-end ML pipeline for medical risk pre-screening using KNN.

    Chains all pipeline stages in sequence with full reproducibility
    (random_state=42 on every stochastic operation).

        load_data()
            └─► preprocess_data(df)
                    └─► train_test_split  (stratified, random_state=42)
                            ├─► train_knn(X_train, y_train)
                            └─► evaluate_model(model, X_test, y_test)

    Returns
    -------
    dict
        'model'         : fitted sklearn Pipeline
        'X_train'       : np.ndarray — training feature matrix
        'X_test'        : np.ndarray — test feature matrix
        'y_train'       : np.ndarray — training labels
        'y_test'        : np.ndarray — test labels
        'scaler'        : StandardScaler (fitted on full data, reference only)
        'label_encoder' : LabelEncoder  (use .inverse_transform() on predictions)
        'metrics'       : dict — accuracy, precision, recall, f1
    """
    print("╔══════════════════════════════════════════════════════╗")
    print("║   KNN Medical Risk Pre-Screening — Pipeline Run     ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Stage 1: Acquire data ────────────────────────────────────────────
    print("── Stage 1: Data Acquisition ──────────────────────────")
    df_raw = load_data()

    # ── Stage 2: Preprocess ──────────────────────────────────────────────
    print("── Stage 2: Preprocessing ─────────────────────────────")
    X_clean, y_encoded, scaler_ref, label_encoder = preprocess_data(df_raw)

    # ── Stage 3: Split ───────────────────────────────────────────────────
    print("── Stage 3: Train/Test Split (80/20, stratified) ──────")
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,  # reproducibility seed
        stratify=y_encoded,         # maintain class proportions in both sets
    )
    print(f"  Train : {X_train.shape[0]} samples")
    print(f"  Test  : {X_test.shape[0]} samples\n")

    # ── Stage 4: Train ───────────────────────────────────────────────────
    print("── Stage 4: Training (sklearn Pipeline) ───────────────")
    model = train_knn(
        X_train, y_train,
        n_neighbors=5,       # initial k — tuned via GridSearchCV in Part 4
        metric="minkowski",  # p=2 default → Euclidean
        weights="uniform",
    )

    # ── Stage 5: Evaluate ────────────────────────────────────────────────
    print("── Stage 5: Evaluation ────────────────────────────────")
    metrics = evaluate_model(model, X_test, y_test, label_encoder)

    print("╔══════════════════════════════════════════════════════╗")
    print("║   Pipeline complete.                                 ║")
    print("╚══════════════════════════════════════════════════════╝")

    return {
        "model"         : model,
        "df_raw"        : df_raw,        # stored so run_sanity_checks can inspect it
        "X"             : X_clean,       # full scaled feature matrix
        "y"             : y_encoded,     # full encoded target
        "X_train"       : X_train,
        "X_test"        : X_test,
        "y_train"       : y_train,
        "y_test"        : y_test,
        "scaler"        : scaler_ref,
        "label_encoder" : label_encoder,
        "metrics"       : metrics,
    }


# ════════════════════════════════════════════════════════════════════════════
# Stage 4b — Hyperparameter Tuning
# ════════════════════════════════════════════════════════════════════════════

def tune_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    scoring: str = "f1_weighted",
    verbose: bool = False,
) -> tuple:
    """
    Tune KNN hyperparameters using GridSearchCV with stratified k-fold CV.

    The search is run over a full Pipeline (StandardScaler → KNN) so that
    the scaler is re-fit on each training fold independently — no leakage.

    Parameter grid
    --------------
    knn__n_neighbors : [3, 5, 7, 9]
        Small k → low bias / high variance.
        Large k → high bias / low variance.
        Odd values used to avoid tie-breaking.

    knn__metric : ['euclidean', 'manhattan']
        euclidean — L2 norm, penalises large deviations quadratically.
        manhattan — L1 norm, linear penalty, more robust to outliers
                    (e.g. Insulin values up to 846).

    knn__weights : ['uniform', 'distance']
        uniform  — each of k neighbours has equal vote weight.
        distance — nearer neighbours vote proportionally more.

    Total combinations : 4 × 2 × 2 = 16
    CV fits per run    : 16 × 5    = 80

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix from train_test_split().
    y_train : np.ndarray
        Integer-encoded training labels.
    cv : int, default=5
        Number of stratified cross-validation folds.
        StratifiedKFold keeps class proportions stable in every fold —
        essential when the 'Low' class has very few training samples (~35).
    scoring : str, default='f1_weighted'
        Optimisation metric. f1_weighted handles class imbalance better
        than accuracy for datasets where classes are not evenly distributed.
    verbose : bool, default=False
        When True, prints per-fold GridSearchCV progress to stdout.

    Returns
    -------
    best_model : sklearn.pipeline.Pipeline
        Pipeline re-fitted on full X_train with optimal hyperparameters.
    best_params : dict
        Best hyperparameter combination found, e.g.
        {'knn__metric': 'manhattan', 'knn__n_neighbors': 7,
         'knn__weights': 'distance'}
    cv_results : pd.DataFrame
        All 16 combinations ranked by mean CV score. Useful for analysing
        sensitivity and trade-offs between parameter settings.
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # ── Step 1: Pipeline to tune ─────────────────────────────────────────────
    # Pipeline wrapping ensures StandardScaler fits only on the training fold,
    # not the validation fold — avoids inflated CV scores from leakage.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn",    KNeighborsClassifier(n_jobs=-1)),
    ])

    # ── Step 2: Hyperparameter search grid ───────────────────────────────────
    # Prefix 'knn__' routes each parameter to the 'knn' Pipeline step.
    param_grid = {
        "knn__n_neighbors": [3, 5, 7, 9],
        "knn__metric"     : ["euclidean", "manhattan"],
        "knn__weights"    : ["uniform", "distance"],
    }

    # ── Step 3: Stratified cross-validation strategy ─────────────────────────
    # shuffle=True with random_state=42 makes fold assignment reproducible.
    # Stratification ensures the minority 'Low' class appears in every fold.
    cv_strategy = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # ── Step 4: GridSearchCV ─────────────────────────────────────────────────
    # refit=True: after ranking, re-trains best pipeline on the full X_train.
    # return_train_score=True: lets us detect over/underfitting per combo.
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=scoring,            # optimise weighted F1
        refit=True,                 # re-fit best model on full X_train
        n_jobs=-1,                  # parallelise CV fits across CPU cores
        verbose=2 if verbose else 0,
        return_train_score=True,
    )

    # ── Step 5: Fit (runs 80 total CV fits) ──────────────────────────────────
    n_combos = (
        len(param_grid["knn__n_neighbors"])
        * len(param_grid["knn__metric"])
        * len(param_grid["knn__weights"])
    )
    print("=" * 60)
    print("HYPERPARAMETER TUNING — GridSearchCV")
    print("=" * 60)
    print(f"  Combinations   : {n_combos}  (4 k × 2 metrics × 2 weights)")
    print(f"  CV folds       : {cv}  (StratifiedKFold, random_state={RANDOM_STATE})")
    print(f"  Total CV fits  : {n_combos * cv}")
    print(f"  Scoring        : {scoring}")
    print(f"  Training on    : {X_train.shape[0]} samples\n")

    grid_search.fit(X_train, y_train)

    # ── Step 6: Extract results ───────────────────────────────────────────────
    best_params = grid_search.best_params_
    best_score  = grid_search.best_score_
    best_model  = grid_search.best_estimator_   # re-fitted on full X_train

    print("=" * 60)
    print("TUNING COMPLETE")
    print("=" * 60)
    print(f"  Best params    : {best_params}")
    print(f"  Best CV score  : {best_score:.4f}  ({scoring})")

    # ── Step 7: Ranked results DataFrame ─────────────────────────────────────
    cv_results = (
        pd.DataFrame(grid_search.cv_results_)
        .filter(regex=r"^(param_|mean_test|std_test|mean_train|rank_test)")
        .sort_values("mean_test_score", ascending=False)
        .reset_index(drop=True)
    )

    display_cols = [
        "param_knn__n_neighbors", "param_knn__metric",
        "param_knn__weights", "mean_test_score", "std_test_score",
    ]
    print("\nTop 10 combinations (mean CV F1-weighted):")
    print(cv_results[display_cols].head(10).to_string(index=False))

    return best_model, best_params, cv_results


# ════════════════════════════════════════════════════════════════════════════
# Stage 5 — Sanity Checks / Debugging
# ════════════════════════════════════════════════════════════════════════════

def run_sanity_checks(
    df_raw: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model: Pipeline,
    label_encoder: LabelEncoder,
) -> dict:
    """
    Assert-based sanity checks across every pipeline artefact.

    Validates shapes, NaN absence, label integrity, split ratios, class
    balance, model trainability, prediction validity, and metric bounds.
    Each failed assert raises AssertionError with an actionable FIX message
    rather than letting errors propagate silently.

    Checks
    ------
     1  X rows == y rows                    (shape alignment)
     2  No NaN in X after preprocessing     (NaN poisons distance calculations)
     3  X has correct number of features    (column count guard)
     4  Dataset >= 500 samples              (fetch completeness)
     5  Target classes == {High,Low,Medium} (label integrity)
     6  Encoded y values in [0, n_classes)  (encoder correctness)
     7  Train + Test == Total samples       (no dropped/duplicated rows)
     8  Test ratio ≈ 20%                    (split ratio sanity)
     9  y_train and y_test contain all classes (stratification check)
    10  Model trains on subsample without exception
    11  Prediction length == X_test rows    (predict() output shape)
    12  All predicted labels are valid      (no out-of-range integers)
    13  Accuracy  in [0, 1]
    14  Precision in [0, 1]
    15  Recall    in [0, 1]
    16  F1        in [0, 1]

    Parameters
    ----------
    df_raw        : raw DataFrame from load_data()
    X             : scaled feature matrix from preprocess_data()
    y             : encoded target vector from preprocess_data()
    X_train       : training feature matrix from train_test_split()
    X_test        : test feature matrix
    y_train       : training labels
    y_test        : test labels
    model         : fitted Pipeline (from train_knn or tune_knn)
    label_encoder : LabelEncoder fitted in preprocess_data()

    Returns
    -------
    dict  {'accuracy', 'precision', 'recall', 'f1'} for the given model on X_test.

    Raises
    ------
    AssertionError  On the first failing check, with label + FIX hint.
    """
    checks_passed = 0
    checks_total  = 0

    def check(condition: bool, label: str, fix: str = "") -> None:
        nonlocal checks_passed, checks_total
        checks_total += 1
        msg = label + (f"\n         FIX → {fix}" if fix else "")
        assert condition, msg          # raises AssertionError with msg if False
        checks_passed += 1
        print(f"  [PASS]  {label}")

    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    # ── 1: Shape alignment ───────────────────────────────────────────────────
    # If X and y have different row counts sklearn will raise a confusing
    # ValueError deep inside fit(); catching it here is far clearer.
    check(
        X.shape[0] == y.shape[0],
        f"X rows ({X.shape[0]}) == y rows ({y.shape[0]})",
        "Re-run preprocess_data() — never drop rows from X or y separately.",
    )

    # ── 2: No NaN in X ───────────────────────────────────────────────────────
    # np.nan propagates through all arithmetic in StandardScaler and through
    # every Euclidean distance calculation, making all predictions nan.
    nan_count = int(np.isnan(X).sum())
    check(
        nan_count == 0,
        f"No NaN in feature matrix  (found {nan_count})",
        "Verify ZERO_AS_NAN_COLS list and median fillna() cover every column.",
    )

    # ── 3: Feature column count ──────────────────────────────────────────────
    check(
        X.shape[1] == len(FEATURE_COLS),
        f"X has {len(FEATURE_COLS)} feature columns  (got {X.shape[1]})",
        f"FEATURE_COLS = {FEATURE_COLS}",
    )

    # ── 4: Minimum sample count ──────────────────────────────────────────────
    # Fewer than 500 rows suggests load_data() received a truncated response.
    check(
        len(df_raw) >= 500,
        f"Dataset >= 500 samples  (got {len(df_raw)})",
        "Check network; re-run load_data().",
    )

    # ── 5: Target class integrity ────────────────────────────────────────────
    expected = {"High", "Low", "Medium"}
    actual   = set(label_encoder.classes_)
    check(
        actual == expected,
        f"Classes == {{High, Low, Medium}}  (got {actual})",
        "Check _assign_risk_label() — it must return only 'High', 'Low', 'Medium'.",
    )

    # ── 6: Encoded label range ───────────────────────────────────────────────
    # Out-of-range integers from LabelEncoder indicate a re-fitting mismatch.
    n_cls = len(label_encoder.classes_)
    check(
        int(y.min()) >= 0 and int(y.max()) < n_cls,
        f"y values in [0, {n_cls-1}]  (min={y.min()}, max={y.max()})",
        "Ensure label_encoder.fit_transform() is called on the Risk column.",
    )

    # ── 7: No lost/duplicated samples in split ───────────────────────────────
    check(
        X_train.shape[0] + X_test.shape[0] == X.shape[0],
        f"Train ({X_train.shape[0]}) + Test ({X_test.shape[0]}) == {X.shape[0]}",
        "Use sklearn train_test_split with the same X from preprocess_data().",
    )

    # ── 8: Approximately 80 / 20 split ───────────────────────────────────────
    ratio = X_test.shape[0] / X.shape[0]
    check(
        0.15 <= ratio <= 0.25,
        f"Test ratio ≈ 20%  (actual {ratio:.1%})",
        f"Verify TEST_SIZE = {TEST_SIZE}.",
    )

    # ── 9: All classes present in both splits ────────────────────────────────
    # The 'Low' class has only ~35 training samples; without stratify=y it
    # can be missing from a split, causing undefined recall for that class.
    for name, arr in [("y_train", y_train), ("y_test", y_test)]:
        found = len(np.unique(arr))
        check(
            found == n_cls,
            f"{name} contains all {n_cls} classes  (found {found})",
            "Pass stratify=y_encoded to train_test_split().",
        )

    # ── 10: Model trains on a subsample without exception ────────────────────
    # A quick smoke-test: 10 samples per class, k=3.
    # Catches dtype mismatches or corrupted data before a full grid search.
    try:
        mini_idx = np.concatenate([
            np.where(y_train == c)[0][:10] for c in np.unique(y_train)
        ])
        smoke = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=3))])
        smoke.fit(X_train[mini_idx], y_train[mini_idx])
        check(True, "Smoke-test fit on 30-sample subsample")
    except Exception as exc:
        check(False, "Smoke-test fit on 30-sample subsample", str(exc))

    # ── 11: Prediction array length ──────────────────────────────────────────
    y_pred = model.predict(X_test)
    check(
        y_pred.shape[0] == X_test.shape[0],
        f"len(y_pred) ({y_pred.shape[0]}) == len(X_test) ({X_test.shape[0]})",
        "Pass the same X_test to both evaluate_model() and model.predict().",
    )

    # ── 12: All predictions are valid class integers ─────────────────────────
    # label_encoder.inverse_transform() raises a cryptic IndexError on
    # out-of-range integers; this check surfaces it with a clear message.
    bad = int(np.sum((y_pred < 0) | (y_pred >= n_cls)))
    check(
        bad == 0,
        f"All predictions are valid integers  ({bad} invalid)",
        "Re-fit label_encoder on the same Risk column used during training.",
    )

    # ── 13–16: Metrics in [0, 1] ─────────────────────────────────────────────
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1w  = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    for name, val in [("Accuracy", acc), ("Precision", prec),
                      ("Recall", rec), ("F1-weighted", f1w)]:
        check(
            0.0 <= val <= 1.0,
            f"{name} in [0, 1]  ({val:.4f})",
            "Use average='weighted', zero_division=0 in sklearn metric calls.",
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  {checks_passed}/{checks_total} checks passed")
    print("=" * 60)
    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (weighted)")
    print(f"  Recall    : {rec:.4f}  (weighted)")
    print(f"  F1-Score  : {f1w:.4f}  (weighted)")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1w}


# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Stage 1-5: baseline pipeline ────────────────────────────────────────
    results = main()

    # ── Stage 6: hyperparameter tuning ──────────────────────────────────────
    print("\n── Stage 6: Hyperparameter Tuning ─────────────────────")
    best_model, best_params, cv_results = tune_knn(
        results["X_train"],
        results["y_train"],
    )

    # ── Stage 7: evaluate the tuned model on the held-out test set ──────────
    print("\n── Stage 7: Tuned Model — Test Set Evaluation ─────────")
    tuned_metrics = evaluate_model(
        best_model,
        results["X_test"],
        results["y_test"],
        results["label_encoder"],
    )

    # ── Baseline vs tuned comparison ────────────────────────────────────────
    le = results["label_encoder"]
    print("\n" + "=" * 60)
    print("BASELINE vs TUNED — Test Set Comparison")
    print("=" * 60)
    comparison = pd.DataFrame(
        {
            "Baseline (k=5, euclidean, uniform)": results["metrics"],
            f"Tuned  {best_params}": tuned_metrics,
        }
    ).T.round(4)
    print(comparison.to_string())

    # ── Stage 8: sanity checks on tuned model ───────────────────────────────
    print("\n── Stage 8: Sanity Checks ─────────────────────────────")
    run_sanity_checks(
        df_raw        = results["df_raw"],
        X             = results["X"],
        y             = results["y"],
        X_train       = results["X_train"],
        X_test        = results["X_test"],
        y_train       = results["y_train"],
        y_test        = results["y_test"],
        model         = best_model,
        label_encoder = results["label_encoder"],
    )

    # ── Sample predictions from tuned model ─────────────────────────────────
    print("\n── Sample Predictions from Tuned Model (first 10) ─────")
    X_test = results["X_test"]
    preds_encoded = best_model.predict(X_test[:10])
    preds_labels  = le.inverse_transform(preds_encoded)
    true_labels   = le.inverse_transform(results["y_test"][:10])
    for i, (true, pred) in enumerate(zip(true_labels, preds_labels)):
        match = "✓" if true == pred else "✗"
        print(f"  Sample {i+1:02d}: true={true:<8s}  pred={pred:<8s}  {match}")

    sys.exit(0)
