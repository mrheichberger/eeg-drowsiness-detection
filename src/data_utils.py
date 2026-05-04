"""
Data loading and metadata utilities.
"""

import os
import zipfile
import shutil
import pandas as pd

from config import CLASS_MAP


def extract_zip(zip_path, extract_path):
    """
    Extract EEG zip file into a clean directory.
    """
    if os.path.exists(extract_path):
        shutil.rmtree(extract_path)

    os.makedirs(extract_path, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    print(f"Extracted data to: {extract_path}")


def safe_read_csv(filepath):
    """
    Read EEG CSV while ignoring comment/header metadata rows.
    """
    df = pd.read_csv(filepath, comment="#")
    df.columns = df.columns.str.strip()
    return df


def build_metadata(base_path):
    """
    Build metadata DataFrame from folder structure.

    Expected structure:
        base_path/
            SubjectName/
                Stimulated/
                Semi/
                rest/
    """
    rows = []

    for subject in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject)

        if not os.path.isdir(subject_path):
            continue

        for raw_class in os.listdir(subject_path):
            class_path = os.path.join(subject_path, raw_class)

            if not os.path.isdir(class_path):
                continue

            if raw_class not in CLASS_MAP:
                print(f"Skipping unknown folder: {class_path}")
                continue

            for filename in os.listdir(class_path):
                if filename.endswith(".csv"):
                    filepath = os.path.join(class_path, filename)

                    rows.append({
                        "subject": subject,
                        "raw_class": raw_class,
                        "class_name": CLASS_MAP[raw_class]["class_name"],
                        "label": CLASS_MAP[raw_class]["label"],
                        "filepath": filepath,
                        "filename": filename,
                    })

    df_meta = pd.DataFrame(rows)

    if len(df_meta) == 0:
        raise ValueError("No EEG CSV files found. Check base_path and folder structure.")

    df_meta = df_meta.sort_values(["subject", "label", "filename"]).reset_index(drop=True)

    return df_meta


def create_loso_splits(df_meta):
    """
    Create Leave-One-Subject-Out splits.
    """
    subjects = sorted(df_meta["subject"].unique())
    loso_splits = []

    for test_subject in subjects:
        train_df = df_meta[df_meta["subject"] != test_subject].reset_index(drop=True)
        test_df = df_meta[df_meta["subject"] == test_subject].reset_index(drop=True)

        loso_splits.append({
            "test_subject": test_subject,
            "train_df": train_df,
            "test_df": test_df,
        })

    return loso_splits


def print_metadata_summary(df_meta):
    """
    Print useful metadata summary.
    """
    print("\nMetadata:")
    print(df_meta)

    print("\nCounts by subject/class:")
    print(df_meta.groupby(["subject", "class_name"]).size())