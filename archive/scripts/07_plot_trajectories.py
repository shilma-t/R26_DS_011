"""
07_plot_trajectories.py
Plots RMS energy and pitch trajectories over time for individual calls.
Shows 5 calls per class so you can see what actually happens during a call,
not just the average.
"""

import os
import random
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CLEAN_DIR   = os.path.join("..", "audio", "clean")
FEATURES_CSV = os.path.join("..", "features.csv")
REPORTS_DIR  = os.path.join("..", "reports")
TARGET_SR    = 16_000
CALLS_PER_CLASS = 5
random.seed(42)


def get_trajectory(wav_path):
    """Load a wav and return time-axis, RMS curve, and voiced F0 curve."""
    y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    hop = 512

    # RMS over time
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times_rms = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    # F0 (pitch) over time — voiced frames only, NaN elsewhere
    f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr, hop_length=hop)
    f0_plot = np.where(voiced, f0, np.nan)
    times_f0 = librosa.frames_to_time(np.arange(len(f0_plot)), sr=sr, hop_length=hop)

    duration = len(y) / sr
    return times_rms, rms, times_f0, f0_plot, duration


def plot_class_trajectories(df, urgency_class, color, ax_rms, ax_pitch):
    calls = df[df["urgency_label"] == urgency_class]["filename"].tolist()
    selected = random.sample(calls, min(CALLS_PER_CLASS, len(calls)))

    for fname in selected:
        stem = os.path.splitext(fname)[0]
        wav  = os.path.join(CLEAN_DIR, stem + ".wav")
        if not os.path.exists(wav):
            continue

        t_rms, rms, t_f0, f0, dur = get_trajectory(wav)

        # Normalise time axis to 0–1 so calls of different lengths are comparable
        t_rms_norm = t_rms / dur
        t_f0_norm  = t_f0  / dur

        ax_rms.plot(t_rms_norm, rms, alpha=0.6, linewidth=1.2, color=color)
        ax_pitch.plot(t_f0_norm, f0,  alpha=0.6, linewidth=1.2, color=color)

    ax_rms.set_title(f"{urgency_class} — RMS Energy", fontsize=11)
    ax_rms.set_ylabel("RMS")
    ax_rms.set_xlabel("Call progress (0=start, 1=end)")

    ax_pitch.set_title(f"{urgency_class} — Pitch (F0)", fontsize=11)
    ax_pitch.set_ylabel("Hz")
    ax_pitch.set_xlabel("Call progress (0=start, 1=end)")


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df = pd.read_csv(FEATURES_CSV)

    classes = ["High", "Medium", "Low"]
    colors  = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}

    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    fig.suptitle(
        f"RMS & Pitch Trajectories — {CALLS_PER_CLASS} calls per class\n"
        "(time normalised: 0 = call start, 1 = call end)",
        fontsize=13, fontweight="bold"
    )

    for row, cls in enumerate(classes):
        plot_class_trajectories(
            df, cls, colors[cls],
            ax_rms=axes[row][0],
            ax_pitch=axes[row][1],
        )

    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, "trajectories_per_class.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

    # ── Also plot class-average trajectory with std band ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Average RMS & Pitch Trajectory per Class (mean ± std band)",
                 fontsize=13, fontweight="bold")

    N_BINS = 20  # divide normalised time into 20 equal bins

    for cls, color in colors.items():
        calls = df[df["urgency_label"] == cls]["filename"].tolist()
        rms_bins   = [[] for _ in range(N_BINS)]
        pitch_bins = [[] for _ in range(N_BINS)]

        for fname in calls:
            stem = os.path.splitext(fname)[0]
            wav  = os.path.join(CLEAN_DIR, stem + ".wav")
            if not os.path.exists(wav):
                continue

            t_rms, rms, t_f0, f0, dur = get_trajectory(wav)
            t_rms_norm = t_rms / dur
            t_f0_norm  = t_f0  / dur

            for b in range(N_BINS):
                lo, hi = b / N_BINS, (b + 1) / N_BINS
                mask_rms = (t_rms_norm >= lo) & (t_rms_norm < hi)
                if mask_rms.any():
                    rms_bins[b].append(np.mean(rms[mask_rms]))

                mask_f0 = (t_f0_norm >= lo) & (t_f0_norm < hi)
                valid_f0 = f0[mask_f0]
                valid_f0 = valid_f0[~np.isnan(valid_f0)]
                if len(valid_f0) > 0:
                    pitch_bins[b].append(np.mean(valid_f0))

        bin_centres = [(b + 0.5) / N_BINS for b in range(N_BINS)]

        # RMS average plot
        ax = axes[0]
        means = [np.mean(v) if v else np.nan for v in rms_bins]
        stds  = [np.std(v)  if v else np.nan for v in rms_bins]
        means = np.array(means)
        stds  = np.array(stds)
        ax.plot(bin_centres, means, color=color, linewidth=2, label=cls)
        ax.fill_between(bin_centres, means - stds, means + stds,
                        color=color, alpha=0.15)

        # Pitch average plot
        ax = axes[1]
        means_p = [np.mean(v) if v else np.nan for v in pitch_bins]
        stds_p  = [np.std(v)  if v else np.nan for v in pitch_bins]
        means_p = np.array(means_p)
        stds_p  = np.array(stds_p)
        ax.plot(bin_centres, means_p, color=color, linewidth=2, label=cls)
        ax.fill_between(bin_centres, means_p - stds_p, means_p + stds_p,
                        color=color, alpha=0.15)

    axes[0].set_title("Average RMS Energy over Call Duration")
    axes[0].set_xlabel("Call progress (0=start, 1=end)")
    axes[0].set_ylabel("RMS")
    axes[0].legend()

    axes[1].set_title("Average Pitch (F0) over Call Duration")
    axes[1].set_xlabel("Call progress (0=start, 1=end)")
    axes[1].set_ylabel("Hz")
    axes[1].legend()

    plt.tight_layout()
    out2 = os.path.join(REPORTS_DIR, "trajectories_average.png")
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
