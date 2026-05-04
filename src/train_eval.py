"""
Training and evaluation functions.
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from config import FS, CHANNELS, WINDOW_SECONDS, STEP_OVERLAP, RANDOM_STATE
from features import build_window_dataset


def make_models():
    """
    Create model dictionary.
    """
    models = {
        "poly_svm": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="poly", degree=3, C=1.0, gamma="scale", coef0=1.0)),
        ]),
        "rbf_svm": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", C=1.0, gamma="scale")),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=RANDOM_STATE,
        ),
    }

    return models


def evaluate_predictions(y_true, y_pred, model_name):
    """
    Return metrics dictionary and print classification report.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n{model_name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }


def run_loso_evaluation(loso_splits):
    """
    Run LOSO evaluation for Poly SVM, RBF SVM, and Random Forest.
    """
    window_size = FS * WINDOW_SECONDS
    step_size = int(window_size * STEP_OVERLAP)

    results = []

    pooled = {
        "poly_svm": {"true": [], "pred": []},
        "rbf_svm": {"true": [], "pred": []},
        "random_forest": {"true": [], "pred": []},
    }

    for fold_idx, split in enumerate(loso_splits, start=1):
        test_subject = split["test_subject"]
        train_df = split["train_df"]
        test_df = split["test_df"]

        print("=" * 70)
        print(f"Fold {fold_idx} | Test subject: {test_subject}")

        X_train, y_train = build_window_dataset(
            train_df,
            channels=CHANNELS,
            fs=FS,
            window_size=window_size,
            step_size=step_size,
            apply_filter=True,
            normalize=True,
        )

        X_test, y_test = build_window_dataset(
            test_df,
            channels=CHANNELS,
            fs=FS,
            window_size=window_size,
            step_size=step_size,
            apply_filter=True,
            normalize=True,
        )

        print("X_train shape:", X_train.shape)
        print("X_test shape: ", X_test.shape)

        models = make_models()

        fold_result = {
            "fold": fold_idx,
            "test_subject": test_subject,
            "n_train_windows": len(X_train),
            "n_test_windows": len(X_test),
        }

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average="macro")

            print(f"{model_name} -> Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}")

            fold_result[f"{model_name}_acc"] = acc
            fold_result[f"{model_name}_f1_macro"] = macro_f1

            pooled[model_name]["true"].extend(y_test.tolist())
            pooled[model_name]["pred"].extend(y_pred.tolist())

        results.append(fold_result)

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("Per-fold LOSO results")
    print(results_df)

    print("\nMean across folds")
    metric_cols = [col for col in results_df.columns if col.endswith("_acc") or col.endswith("_f1_macro")]
    print(results_df[metric_cols].mean())

    print("\n" + "=" * 70)
    print("Overall pooled results")

    for model_name in pooled:
        evaluate_predictions(
            pooled[model_name]["true"],
            pooled[model_name]["pred"],
            model_name,
        )

    return results_df, pooled


def run_time_split_evaluation(
    df_meta,
    subjects_to_use,
    purge_seconds=2,
):
    """
    Run time-based 80/20 split with purge gap.
    This is useful for testing within-subject performance.
    """
    from data_utils import safe_read_csv
    from features import bandpass_filter_signal, normalize_signal_per_file, create_windows, extract_features

    window_size = FS * WINDOW_SECONDS
    step_size = int(window_size * STEP_OVERLAP)
    purge_samples = int(purge_seconds * FS)

    df_subset = df_meta[df_meta["subject"].isin(subjects_to_use)].reset_index(drop=True)

    X_train_all = []
    y_train_all = []
    X_test_all = []
    y_test_all = []

    for _, row in df_subset.iterrows():
        filepath = row["filepath"]
        label = row["label"]

        try:
            df = safe_read_csv(filepath)

            if not all(ch in df.columns for ch in CHANNELS):
                print(f"Skipping {filepath}: missing channels.")
                continue

            signal = df[CHANNELS].to_numpy(dtype=float)

            signal = bandpass_filter_signal(signal, FS, low=0.5, high=40)
            signal = normalize_signal_per_file(signal)

            n_samples = len(signal)
            split_idx = int(0.8 * n_samples)

            train_signal = signal[:split_idx - purge_samples]
            test_signal = signal[split_idx + purge_samples:]

            if len(train_signal) < window_size or len(test_signal) < window_size:
                continue

            train_windows = create_windows(train_signal, window_size, step_size)
            test_windows = create_windows(test_signal, window_size, step_size)

            for window in train_windows:
                X_train_all.append(extract_features(window, fs=FS))
                y_train_all.append(label)

            for window in test_windows:
                X_test_all.append(extract_features(window, fs=FS))
                y_test_all.append(label)

        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    X_train_all = np.array(X_train_all)
    y_train_all = np.array(y_train_all)
    X_test_all = np.array(X_test_all)
    y_test_all = np.array(y_test_all)

    print("Train shape:", X_train_all.shape)
    print("Test shape: ", X_test_all.shape)

    models = make_models()
    trained_models = {}

    for model_name, model in models.items():
        model.fit(X_train_all, y_train_all)
        y_pred = model.predict(X_test_all)

        evaluate_predictions(y_test_all, y_pred, model_name)
        trained_models[model_name] = model

    return trained_models, X_test_all, y_test_all


def save_models(models, model_dir):
    """
    Save trained models using joblib.
    """
    os.makedirs(model_dir, exist_ok=True)

    for model_name, model in models.items():
        model_path = os.path.join(model_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        print(f"Saved {model_name} to {model_path}")