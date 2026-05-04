"""
Visualization helpers for EEG analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from sklearn.metrics import confusion_matrix

from config import FS, CHANNELS
from data_utils import safe_read_csv
from features import bandpower


def plot_confusion_matrix(y_true, y_pred, title):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1, 2], ["Drowsy", "Mid", "Stimulated"])
    plt.yticks([0, 1, 2], ["Drowsy", "Mid", "Stimulated"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.show()


def plot_normalized_psd_by_subject(df_meta):
    """
    Plot normalized PSD by subject and class.
    """
    class_order = ["stimulated", "mid", "drowsy"]
    fmin, fmax = 0.5, 40

    subject_class_psds = {}
    freqs_band = None

    for _, row in df_meta.iterrows():
        filepath = row["filepath"]
        subject = row["subject"]
        class_name = row["class_name"]

        try:
            df = safe_read_csv(filepath)

            if not all(ch in df.columns for ch in CHANNELS):
                continue

            signal = df[CHANNELS].to_numpy(dtype=float)

            psds = []
            freqs_ref = None

            for ch_idx in range(len(CHANNELS)):
                f, pxx = welch(signal[:, ch_idx], fs=FS, nperseg=1024)

                if freqs_ref is None:
                    freqs_ref = f

                psds.append(pxx)

            mean_psd = np.mean(psds, axis=0)

            band_mask = (freqs_ref >= fmin) & (freqs_ref <= fmax)
            freqs_band = freqs_ref[band_mask]
            psd_band = mean_psd[band_mask]

            total_power = np.trapezoid(psd_band, freqs_band)

            if total_power > 0:
                psd_norm = psd_band / total_power
            else:
                psd_norm = psd_band

            subject_class_psds.setdefault(subject, {})
            subject_class_psds[subject].setdefault(class_name, [])
            subject_class_psds[subject][class_name].append(psd_norm)

        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    subjects = sorted(subject_class_psds.keys())

    fig, axes = plt.subplots(1, len(subjects), figsize=(6 * len(subjects), 5), sharex=True, sharey=True)

    if len(subjects) == 1:
        axes = [axes]

    global_ymax = 0
    subject_means = {}

    for subject in subjects:
        subject_means[subject] = {}

        for class_name in class_order:
            if class_name in subject_class_psds[subject]:
                class_psd_mean = np.mean(subject_class_psds[subject][class_name], axis=0)
                subject_means[subject][class_name] = class_psd_mean
                global_ymax = max(global_ymax, np.max(class_psd_mean))

    for ax, subject in zip(axes, subjects):
        for class_name in class_order:
            if class_name in subject_means[subject]:
                ax.plot(freqs_band, subject_means[subject][class_name], label=class_name)

        ax.set_title(subject)
        ax.set_xlim(fmin, fmax)
        ax.set_ylim(0, global_ymax * 1.05)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Relative Power")
        ax.grid(True)
        ax.legend()

    plt.suptitle("Normalized PSD by Subject")
    plt.tight_layout()
    plt.show()


def build_bandpower_feature_table(df_meta):
    """
    Build per-channel bandpower feature table for visualization.
    """
    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 13)
    beta_band = (13, 30)

    eps = 1e-12
    rows = []

    for _, row in df_meta.iterrows():
        filepath = row["filepath"]
        subject = row["subject"]
        class_name = row["class_name"]
        filename = row["filename"]

        try:
            df = safe_read_csv(filepath)

            if not all(ch in df.columns for ch in CHANNELS):
                continue

            signal = df[CHANNELS].to_numpy(dtype=float)

            for ch_idx, ch_name in enumerate(CHANNELS):
                x = signal[:, ch_idx]

                delta = bandpower(x, FS, delta_band)
                theta = bandpower(x, FS, theta_band)
                alpha = bandpower(x, FS, alpha_band)
                beta = bandpower(x, FS, beta_band)

                rows.append({
                    "subject": subject,
                    "class_name": class_name,
                    "filename": filename,
                    "channel": ch_name,
                    "delta": delta,
                    "theta": theta,
                    "alpha": alpha,
                    "beta": beta,
                    "theta_alpha": theta / (alpha + eps),
                    "theta_beta": theta / (beta + eps),
                    "alpha_beta": alpha / (beta + eps),
                })

        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    return pd.DataFrame(rows)


def plot_bandpower_boxplots(feature_df, group_by="class_name"):
    """
    Plot bandpower and ratio boxplots grouped by class or subject.
    """
    features_to_plot = ["delta", "theta", "alpha", "beta", "theta_alpha", "theta_beta", "alpha_beta"]

    feature_titles = {
        "delta": "Delta Power",
        "theta": "Theta Power",
        "alpha": "Alpha Power",
        "beta": "Beta Power",
        "theta_alpha": "Theta / Alpha",
        "theta_beta": "Theta / Beta",
        "alpha_beta": "Alpha / Beta",
    }

    groups = sorted(feature_df[group_by].unique())

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    axes = axes.flatten()

    for i, feat in enumerate(features_to_plot):
        ax = axes[i]

        data = [
            feature_df.loc[feature_df[group_by] == group, feat].dropna().values
            for group in groups
        ]

        ax.boxplot(data, tick_labels=groups)
        ax.set_title(feature_titles[feat])
        ax.set_xlabel(group_by)
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)

    for j in range(len(features_to_plot), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Bandpower Features Grouped by {group_by}")
    plt.tight_layout()
    plt.show()