"""
Configuration settings for EEG drowsiness detection project.
"""

FS = 250

CHANNELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

CLASS_MAP = {
    "Stimulated": {"class_name": "stimulated", "label": 2},
    "Semi": {"class_name": "mid", "label": 1},
    "rest": {"class_name": "drowsy", "label": 0},
}

LABEL_TO_STATE = {
    2: "stimulated",
    1: "mid",
    0: "drowsy",
}

STATE_TO_LIGHT = {
    "stimulated": "BRIGHT",
    "mid": "DIM",
    "drowsy": "DARK",
}

WINDOW_SECONDS = 4
STEP_OVERLAP = 0.5

RANDOM_STATE = 42