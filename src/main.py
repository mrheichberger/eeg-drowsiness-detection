#-----------------------
#Main Source Code 
#-----------------------

"""
Main script for EEG drowsiness detection project.

Example usage:
    python src/main.py
"""

import os
import numpy as np

from config import FS, WINDOW_SECONDS, STEP_OVERLAP
from data_utils import extract_zip, build_metadata, create_loso_splits, print_metadata_summary
from train_eval import run_loso_evaluation, run_time_split_evaluation, save_models


def main():
    zip_path = "../data/EEG_Project_Data.zip"
    extract_path = "../data/EEG_data/EEG_Project_Data (3)"
    model_dir = "../models"


    should_extract_zip = False

    if should_extract_zip:
        extract_zip(zip_path, extract_path)

    base_path = extract_path

    print("Building metadata...")
    df_meta = build_metadata(base_path)
    print_metadata_summary(df_meta)

    print("\nCreating LOSO splits...")
    loso_splits = create_loso_splits(df_meta)

    print("\nRunning LOSO evaluation...")
    loso_results_df, pooled = run_loso_evaluation(loso_splits)

    print("\nRunning time-based 80/20 split evaluation...")
    subjects_to_use = ["Coop", "Paul"]

    trained_models, X_test_all, y_test_all = run_time_split_evaluation(
        df_meta,
        subjects_to_use=subjects_to_use,
        purge_seconds=2,
    )

    print("\nSaving trained time-split models...")
    save_models(trained_models, model_dir)

    os.makedirs("exports", exist_ok=True)
    np.save("exports/X_test_all.npy", X_test_all)
    np.save("exports/y_test_all.npy", y_test_all)

    print("\nSaved test feature arrays to exports/.")


if __name__ == "__main__":
    main()