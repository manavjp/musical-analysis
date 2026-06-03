"""
Audio Analysis Script
Song: Nagari Nagari Dware Dware — Mother India (1957)
Singer: Lata Mangeshkar | Music: Naushad | Lyrics: Shakeel Badayuni
"""

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ── Configuration ──────────────────────────────────────────────────────────────
AUDIO_FILE = "nagri_nagri.mp3"
OUTPUT_DIR = "analysis_plots"
TITLE_INFO = "Nagari Nagari Dware Dware  |  Mother India (1957)  |  Lata Mangeshkar"
DPI = 150

os.makedirs(OUTPUT_DIR, exist_ok=True)

DARK_BG   = "#0d0d0d"
ACCENT    = "#e8c97a"   # warm gold
ACCENT2   = "#7ab8e8"   # cool blue
ACCENT3   = "#e87a7a"   # red
ACCENT4   = "#7ae87a"   # green
TEXT      = "#f0e8d8"

def fig_header(fig, subtitle):
    fig.text(0.5, 0.98, TITLE_INFO, ha="center", va="top",
             fontsize=9, color=ACCENT, style="italic")
    fig.text(0.5, 0.955, subtitle, ha="center", va="top",
             fontsize=13, color=TEXT, weight="bold")

def style_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor("#161616")
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333333")
    if xlabel: ax.set_xlabel(xlabel, color=TEXT, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT, fontsize=8)
    if title:  ax.set_title(title, color=ACCENT, fontsize=9, pad=4)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight",
                facecolor=DARK_BG, edgecolor="none")
    plt.close(fig)
    print(f"  saved → {path}")

# ── Load audio ─────────────────────────────────────────────────────────────────
print("Loading audio …")
y, sr = librosa.load(AUDIO_FILE, sr=None)
duration = librosa.get_duration(y=y, sr=sr)
print(f"  Duration : {duration:.1f} s  |  SR : {sr} Hz  |  Samples : {len(y):,}")

# ── Pre-compute all features ───────────────────────────────────────────────────
print("Computing features …")

t_samples = np.linspace(0, duration, len(y))

# RMS energy
rms          = librosa.feature.rms(y=y)[0]
rms_t        = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

# STFT
D            = librosa.stft(y)
D_db         = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Mel spectrogram
mel_spec     = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_db       = librosa.power_to_db(mel_spec, ref=np.max)

# MFCCs (20 coefficients)
mfccs        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

# Chroma
chroma       = librosa.feature.chroma_stft(y=y, sr=sr)

# Spectral features
spec_centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
spec_t         = librosa.frames_to_time(np.arange(len(spec_centroid)), sr=sr)

# Onset detection
onset_env   = librosa.onset.onset_strength(y=y, sr=sr)
onset_t     = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr)
onsets      = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="time")
tempo_val, beats = librosa.beat.beat_track(y=y, sr=sr)
beat_times  = librosa.frames_to_time(beats, sr=sr)
if isinstance(tempo_val, np.ndarray):
    tempo_val = float(tempo_val.item())

# Tempogram
tempogram   = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)

# Pitch tracking (pyin — most accurate for vocals)
f0, voiced_flag, voiced_prob = librosa.pyin(
    y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
)
pitch_t = librosa.frames_to_time(np.arange(len(f0)), sr=sr)

print("  All features ready.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Time-domain overview
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 1: Time-domain …")
fig, axes = plt.subplots(2, 1, figsize=(14, 7), facecolor=DARK_BG)
fig.subplots_adjust(hspace=0.45)
fig_header(fig, "Figure 1 — Time-Domain Features")

ax = axes[0]
ax.fill_between(t_samples, y, color=ACCENT, alpha=0.7, linewidth=0)
ax.axhline(0, color="#444", linewidth=0.5)
style_ax(ax, xlabel="Time (s)", ylabel="Amplitude", title="Waveform")
ax.set_xlim(0, duration)

ax = axes[1]
ax.plot(rms_t, rms, color=ACCENT2, linewidth=0.9)
ax.fill_between(rms_t, rms, color=ACCENT2, alpha=0.3)
loud_idx  = np.argmax(rms)
quiet_idx = np.argmin(rms)
ax.axvline(rms_t[loud_idx],  color=ACCENT3, linewidth=1, linestyle="--", label=f"Loudest  {rms_t[loud_idx]:.1f}s")
ax.axvline(rms_t[quiet_idx], color=ACCENT4,  linewidth=1, linestyle="--", label=f"Quietest {rms_t[quiet_idx]:.1f}s")
ax.legend(fontsize=7, facecolor="#222", labelcolor=TEXT, loc="upper right")
style_ax(ax, xlabel="Time (s)", ylabel="RMS", title="RMS Energy  (overall loudness per frame)")
ax.set_xlim(0, duration)

save(fig, "fig1_time_domain.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Spectrograms (STFT, Mel)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 2: Spectrograms …")
fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=DARK_BG)
fig.subplots_adjust(hspace=0.5)
fig_header(fig, "Figure 2 — Frequency Spectrograms")

ax = axes[0]
img = librosa.display.specshow(D_db, sr=sr, x_axis="time", y_axis="hz",
                               ax=ax, cmap="inferno")
fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01).ax.yaxis.set_tick_params(labelcolor=TEXT)
style_ax(ax, title="STFT Spectrogram  (linear frequency scale, dB)")

ax = axes[1]
img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel",
                               ax=ax, cmap="magma")
fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01).ax.yaxis.set_tick_params(labelcolor=TEXT)
style_ax(ax, title="Mel Spectrogram  (perceptual frequency scale, dB)")

save(fig, "fig2_spectrograms.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Tonal / harmonic features
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 3: Tonal features …")
fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=DARK_BG)
fig.subplots_adjust(hspace=0.5)
fig_header(fig, "Figure 3 — Tonal & Harmonic Features")

PITCH_CLASSES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

ax = axes[0]
img = librosa.display.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma",
                               ax=ax, cmap="plasma")
fig.colorbar(img, ax=ax, pad=0.01).ax.yaxis.set_tick_params(labelcolor=TEXT)
style_ax(ax, title="Chromagram  (pitch-class energy over time — shows raga/scale activity)")

ax = axes[1]
img = librosa.display.specshow(mfccs, sr=sr, x_axis="time", ax=ax, cmap="coolwarm")
ax.set_yticks(np.arange(20))
ax.set_yticklabels([f"MFCC {i+1}" for i in range(20)], fontsize=6)
fig.colorbar(img, ax=ax, pad=0.01).ax.yaxis.set_tick_params(labelcolor=TEXT)
style_ax(ax, title="MFCCs  (timbre / vocal quality fingerprint — 20 coefficients)")

save(fig, "fig3_tonal_mfcc.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Spectral statistics over time
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 4: Spectral statistics …")
fig, axes = plt.subplots(2, 1, figsize=(14, 7), facecolor=DARK_BG)
fig.subplots_adjust(hspace=0.5)
fig_header(fig, "Figure 4 — Spectral Statistics Over Time")

rms_db = librosa.amplitude_to_db(rms, ref=np.max)

datasets = [
    (spec_centroid,  ACCENT,  "Spectral Centroid (Hz)",
     "Spectral Centroid  (brightness — higher = brighter/sharper sound)"),
    (spec_bandwidth, ACCENT2, "Bandwidth (Hz)",
     "Spectral Bandwidth  (spread of frequencies — wider = richer texture)"),
]
for ax, (data, color, ylabel, title) in zip(axes, datasets):
    ax.plot(spec_t, data, color=color, linewidth=0.8)
    ax.fill_between(spec_t, data, alpha=0.25, color=color)
    win = max(1, len(data) // 80)
    smoothed = np.convolve(data, np.ones(win)/win, mode="same")
    ax.plot(spec_t, smoothed, color="white", linewidth=1.2, alpha=0.6, linestyle="--")
    ax.set_xlim(0, duration)
    style_ax(ax, xlabel="Time (s)", ylabel=ylabel, title=title)

save(fig, "fig4_spectral_stats.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Rhythm & tempo
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 5: Rhythm & tempo …")
fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=DARK_BG)
fig.subplots_adjust(hspace=0.5)
fig_header(fig, f"Figure 5 — Rhythm & Tempo  (estimated BPM: {tempo_val:.1f})")

ax = axes[0]
ax.plot(onset_t, onset_env, color=ACCENT2, linewidth=0.8, label="Onset strength")
for bt in beat_times:
    ax.axvline(bt, color=ACCENT, linewidth=0.5, alpha=0.6)
for ot in onsets:
    ax.axvline(ot, color=ACCENT3, linewidth=0.4, alpha=0.35)
ax.set_xlim(0, duration)
ax.legend(["Onset strength",
           f"Beat positions (n={len(beat_times)})",
           f"Onset events (n={len(onsets)})"],
          fontsize=7, facecolor="#222", labelcolor=TEXT)
style_ax(ax, xlabel="Time (s)", ylabel="Strength",
         title="Onset Strength  (gold lines = beats, red ticks = detected onsets)")

ax = axes[1]
img = librosa.display.specshow(tempogram, sr=sr, x_axis="time",
                               y_axis="tempo", cmap="hot", ax=ax)
ax.axhline(tempo_val, color=ACCENT2, linewidth=1, linestyle="--",
           label=f"Estimated tempo {tempo_val:.1f} BPM")
ax.legend(fontsize=7, facecolor="#222", labelcolor=TEXT)
fig.colorbar(img, ax=ax, pad=0.01).ax.yaxis.set_tick_params(labelcolor=TEXT)
style_ax(ax, title="Tempogram  (how tempo energy is distributed over time)")

save(fig, "fig5_rhythm_tempo.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Pitch (f0) tracking
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 6: Pitch tracking …")
fig, axes = plt.subplots(2, 1, figsize=(14, 8), facecolor=DARK_BG)
fig.subplots_adjust(hspace=0.45)
fig_header(fig, "Figure 6 — Fundamental Frequency (Pitch) Tracking  [pYIN]")

voiced_f0 = np.where(voiced_flag, f0, np.nan)
unvoiced_f0 = np.where(~voiced_flag, f0, np.nan)

ax = axes[0]
ax.scatter(pitch_t, voiced_f0, s=0.5, c=ACCENT, alpha=0.7, label="Voiced (melody)")
ax.scatter(pitch_t, unvoiced_f0, s=0.3, c="#555", alpha=0.4, label="Unvoiced")
ax2 = ax.twinx()
ax2.fill_between(pitch_t, voiced_prob, color=ACCENT2, alpha=0.15, label="Voiced probability")
ax2.set_ylim(0, 1.2)
ax2.tick_params(colors=TEXT, labelsize=7)
ax2.set_ylabel("Voiced probability", color=ACCENT2, fontsize=8)
for note, hz in [("E4",329.6),("A4",440),("E5",659.3),("A5",880)]:
    ax.axhline(hz, color="#444", linewidth=0.5, linestyle=":")
    ax.text(duration*0.01, hz+5, note, color="#888", fontsize=6)
ax.set_xlim(0, duration)
ax.set_ylim(50, 2000)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.legend(fontsize=7, facecolor="#222", labelcolor=TEXT, loc="upper right")
style_ax(ax, xlabel="Time (s)", ylabel="Frequency (Hz, log)",
         title="F0 Pitch Contour  (melody line — log scale, dotted lines = reference notes)")

ax = axes[1]
voiced_only = voiced_f0[~np.isnan(voiced_f0)]
if len(voiced_only):
    notes = librosa.hz_to_midi(voiced_only)
    ax.hist(notes, bins=88, range=(36, 96), color=ACCENT, edgecolor="#222", alpha=0.85)
    ax.set_xticks(np.arange(36, 97, 12))
    ax.set_xticklabels([librosa.midi_to_note(m) for m in range(36, 97, 12)], fontsize=7)
style_ax(ax, xlabel="MIDI note", ylabel="Frame count",
         title="Pitch Histogram  (note distribution across the entire song — voiced frames only)")

save(fig, "fig6_pitch.png")

# ── Done ───────────────────────────────────────────────────────────────────────
print("\nAll figures saved to:", os.path.abspath(OUTPUT_DIR))
print(f"\n{'─'*55}")
print(f"  Song stats summary")
print(f"{'─'*55}")
print(f"  Duration            : {duration:.1f} s  ({duration/60:.1f} min)")
print(f"  Sample rate         : {sr} Hz")
print(f"  Estimated tempo     : {tempo_val:.1f} BPM")
print(f"  Beat count          : {len(beat_times)}")
print(f"  Onset events        : {len(onsets)}")
print(f"  Dominant pitch class: {PITCH_CLASSES[chroma.mean(axis=1).argmax()]}")
if len(voiced_only):
    print(f"  Pitch range (voiced): {voiced_only.min():.0f} – {voiced_only.max():.0f} Hz")
    print(f"  Median pitch        : {np.nanmedian(voiced_f0):.0f} Hz  "
          f"({librosa.hz_to_note(np.nanmedian(voiced_f0))})")
_rms_db = librosa.amplitude_to_db(rms, ref=np.max)
print(f"  Mean spectral cent  : {spec_centroid.mean():.0f} Hz")
print(f"  Dynamic range (RMS) : {_rms_db.max():.1f} – {_rms_db.min():.1f} dB")
print(f"{'─'*55}")
