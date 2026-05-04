"""
EEG Light Classification Demo Server
"""

import asyncio
import json
from pathlib import Path

import joblib
import numpy as np
import websockets


MODEL_PATH = Path("poly_svm_time_split.joblib")
X_TEST_PATH = Path("X_test_all.npy")
Y_TEST_PATH = Path("y_test_all.npy")

HOST = "localhost"
PORT = 8765

PREDICTION_INTERVAL_SECONDS = 0.5
SUSTAINED_WINDOWS_REQUIRED = 5

# 2 = stimulated, 1 = mid, 0 = drowsy
label_to_state = {
    2: "stimulated",
    1: "mid",
    0: "drowsy",
}

state_to_brightness = {
    "stimulated": 100,
    "mid": 50,
    "drowsy": 10,
}


def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    if not X_TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test feature file: {X_TEST_PATH}")

    model = joblib.load(MODEL_PATH)
    X_test = np.load(X_TEST_PATH)

    y_test = None
    if Y_TEST_PATH.exists():
        y_test = np.load(Y_TEST_PATH)

    return model, X_test, y_test


def update_sustained_state(raw_label, candidate_label, candidate_count, applied_label):
    if raw_label == candidate_label:
        candidate_count += 1
    else:
        candidate_label = raw_label
        candidate_count = 1

    changed_state = False

    if candidate_count >= SUSTAINED_WINDOWS_REQUIRED and applied_label != candidate_label:
        applied_label = candidate_label
        changed_state = True

    if applied_label is None:
        applied_label = raw_label

    return candidate_label, candidate_count, applied_label, changed_state


async def prediction_stream(websocket):
    print("HTML page connected.")

    model, X_test, y_test = load_assets()

    correct = 0
    total = 0

    candidate_label = None
    candidate_count = 0
    applied_label = None

    for i, features in enumerate(X_test):
        raw_label = int(model.predict(features.reshape(1, -1))[0])
        raw_state = label_to_state.get(raw_label, "unknown")

        candidate_label, candidate_count, applied_label, changed_state = update_sustained_state(
            raw_label=raw_label,
            candidate_label=candidate_label,
            candidate_count=candidate_count,
            applied_label=applied_label,
        )

        candidate_state = label_to_state.get(candidate_label, "unknown")
        applied_state = label_to_state.get(applied_label, "unknown")
        brightness = state_to_brightness.get(applied_state, 0)

        message = {
            "index": i,

            "raw_label": raw_label,
            "raw_state": raw_state,

            "candidate_label": candidate_label,
            "candidate_state": candidate_state,
            "candidate_count": candidate_count,
            "required_count": SUSTAINED_WINDOWS_REQUIRED,
            "changed_state": changed_state,

	    # model prediction shown in HTML
	    "label": raw_label,
	    "state": raw_state,

	    # sustained/smoothed state used only for light brightness
	    "light_label": applied_label,
	    "light_state": applied_state,
	    "brightness": brightness,

        }

        if y_test is not None and i < len(y_test):
            true_label = int(y_test[i])
            true_state = label_to_state.get(true_label, "unknown")

            message["true_label"] = true_label
            message["true_state"] = true_state

            total += 1
            if raw_label == true_label:
                correct += 1

            message["running_accuracy"] = correct / total

        await websocket.send(json.dumps(message))
        print(message)

        await asyncio.sleep(PREDICTION_INTERVAL_SECONDS)

    print("Finished streaming all test windows.")


async def main():
    print(f"Starting WebSocket server at ws://{HOST}:{PORT}")
    print("Open index.html in your browser after this starts.")
    print(f"Prediction interval: {PREDICTION_INTERVAL_SECONDS} seconds")
    print(f"Sustained windows required: {SUSTAINED_WINDOWS_REQUIRED}")

    async with websockets.serve(prediction_stream, HOST, PORT):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())