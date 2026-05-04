"""
Signal processing, windowing, and feature extraction.
"""

import numpy as np
from scipy.signal import butter, filtfilt, welch

from data_utils import safe_read_csv


def bandpass_filter_signal(signal_2d, fs, low=0.5, high=40, order=4):
    """
    Apply Butterworth bandpass filter to EEG signal.

    signal_2d shape:
        samples x channels
    """
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal_2d, axis=0)


def create_windows(signal_2d, window_size, step_size):
    """
    Create sliding windows from EEG signal.
    """
    windows = []

    for start in range(0, len(signal_2d) - window_size + 1, step_size):
        end = start + window_size
        windows.append(signal_2d[start:end])

    return np.array(windows)


def bandpower(x, fs, band):
    """
    Compute bandpower using Welch PSD estimate.
    """
    low, high = band
    freqs, psd = welch(x, fs=fs, nperseg=min(256, len(x)))

    idx = (freqs >= low) & (freqs <= high)

    if np.sum(idx) == 0:
        return 0.0

    return np.trapezoid(psd[idx], freqs[idx])


def extract_features(window, fs=250):
    """
    Extract EEG features from one window.

    Features per channel:
        mean
        std
        variance
        delta power
        theta power
        alpha power
        beta power
        relative delta
        relative theta
        relative alpha
        relative beta
        theta/alpha
        theta/beta
        alpha/beta
    """
    features = []

    delta = (0.5, 4)
    theta = (4, 8)
    alpha = (8, 13)
    beta = (13, 30)

    eps = 1e-12

    for ch in range(window.shape[1]):
        sig_ch = window[:, ch]

        mean_val = np.mean(sig_ch)
        std_val = np.std(sig_ch)
        var_val = np.var(sig_ch)

        delta_power = bandpower(sig_ch, fs, delta)
        theta_power = bandpower(sig_ch, fs, theta)
        alpha_power = bandpower(sig_ch, fs, alpha)
        beta_power = bandpower(sig_ch, fs, beta)

        total_power = delta_power + theta_power + alpha_power + beta_power + eps

        rel_delta = delta_power / total_power
        rel_theta = theta_power / total_power
        rel_alpha = alpha_power / total_power
        rel_beta = beta_power / total_power

        theta_alpha = theta_power / (alpha_power + eps)
        theta_beta = theta_power / (beta_power + eps)
        alpha_beta = alpha_power / (beta_power + eps)

        features.extend([
            mean_val,
            std_val,
            var_val,
            delta_power,
            theta_power,
            alpha_power,
            beta_power,
            rel_delta,
            rel_theta,
            rel_alpha,
            rel_beta,
            theta_alpha,
            theta_beta,
            alpha_beta,
        ])

    return np.array(features, dtype=float)


def normalize_signal_per_file(signal_2d):
    """
    Normalize each recording independently.
    This removes per-recording scale/baseline differences.
    """
    mu = signal_2d.mean(axis=0, keepdims=True)
    sigma = signal_2d.std(axis=0, keepdims=True) + 1e-8
    return (signal_2d - mu) / sigma


def build_window_dataset(
    meta_df,
    channels,
    fs,
    window_size,
    step_size,
    apply_filter=True,
    normalize=True,
):
    """
    Convert file-level metadata into window-level feature dataset.
    """
    X = []
    y = []

    for _, row in meta_df.iterrows():
        filepath = row["filepath"]
        label = row["label"]

        try:
            df = safe_read_csv(filepath)

            if not all(ch in df.columns for ch in channels):
                print(f"Skipping {filepath}: missing required channels.")
                continue

            signal_2d = df[channels].to_numpy(dtype=float)

            if apply_filter:
                signal_2d = bandpass_filter_signal(signal_2d, fs, low=0.5, high=40)

            if normalize:
                signal_2d = normalize_signal_per_file(signal_2d)

            windows = create_windows(signal_2d, window_size, step_size)

            for window in windows:
                X.append(extract_features(window, fs=fs))
                y.append(label)

        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    return np.array(X), np.array(y)