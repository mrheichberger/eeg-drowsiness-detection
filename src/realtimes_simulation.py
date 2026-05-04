"""
Real-time classification simulation.

This uses precomputed feature windows, such as X_test_all.npy.
"""

import time
import joblib
import numpy as np

from config import LABEL_TO_STATE, STATE_TO_LIGHT


def run_realtime_classification(
    model_path,
    data_windows,
    ground_truth_labels=None,
    interval_seconds=0.1,
    sustained_count=1,
):
    """
    Simulate real-time EEG classification.

    Parameters:
        model_path: path to .joblib model
        data_windows: feature matrix, shape n_windows x n_features
        ground_truth_labels: optional true labels
        interval_seconds: pause between predictions
        sustained_count: number of repeated predictions needed before changing light
    """
    print(f"Loading model from: {model_path}")

    model = joblib.load(model_path)
    print("Model loaded successfully.")

    previous_label = None
    repeated_count = 0
    active_state = None

    num_correct = 0
    total_predictions = 0

    print("\nStarting real-time classification simulation.")
    print("Press Ctrl+C to stop.\n")

    try:
        for i, window_features in enumerate(data_windows):
            predicted_label = model.predict(window_features.reshape(1, -1))[0]

            if predicted_label == previous_label:
                repeated_count += 1
            else:
                repeated_count = 1
                previous_label = predicted_label

            predicted_state = LABEL_TO_STATE.get(predicted_label, "unknown")

            if repeated_count >= sustained_count:
                active_state = predicted_state

            light_state = STATE_TO_LIGHT.get(active_state, "UNKNOWN")

            output = (
                f"Window {i + 1}: "
                f"Predicted={predicted_label} ({predicted_state}) | "
                f"Light={light_state}"
            )

            if ground_truth_labels is not None:
                true_label = ground_truth_labels[i]
                true_state = LABEL_TO_STATE.get(true_label, "unknown")

                output += f" | True={true_label} ({true_state})"

                if predicted_label == true_label:
                    num_correct += 1

                total_predictions += 1

            print(output)
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

    if total_predictions > 0:
        accuracy = num_correct / total_predictions
        print(f"\nSimulation accuracy: {accuracy:.4f}")