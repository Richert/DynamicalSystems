"""Compare stored spike events against OASIS deconvolution output.

Loads the calcium traces and the stored spike events for one session
(a given drug/dose/mouse/condition) and, for a few example neurons, overlays:
  - the input calcium trace,
  - the stored (binary) spike events, and
  - the full, non-thresholded OASIS deconvolution output `s` for both the
    L1 and L0 sparsity penalties.

OASIS `s` is a per-frame non-negative activity estimate (proportional to the
spike count in each frame), not a list of spike times. L1 gives a graded
output; L0 gives a sparser one. Here we plot `s` directly (no thresholding).

Run with the dedicated env:
  /home/rgast/conda/envs/spikes/bin/python compare_spike_extraction.py
"""

import os
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

# session selection
path = "/home/rgast/data/parker_data/neural_data"
drug = "haloperidol"
dose = "Vehicle"          # "Vehicle" / "LowDose" / "HighDose"
mouse = "m085"            # substring matched against the session folder name
condition = "amph"        # "veh" or "amph"

# input trace: "dff_traces_5hz" (dF/F) or "cnmfe_traces_5hz" (CNMF-E denoised).
# When 20 Hz data becomes available, point this at the 20 Hz field and set fs=20.
input_field = "cnmfe_traces_5hz"
events_field = "events_5hz"
fs = 5.0                 # sampling rate of the input traces (Hz)

# OASIS: AR model order, applied for both penalties
ar_order = 1             # 1 (AR1) or 2 (AR2)
penalties = {"L1": 1, "L0": 0}   # label -> OASIS penalty argument

# --- preprocessing options (applied to the input traces before extraction) ---
preprocessing = {
    "detrend_percentile": None,  # e.g. 8 -> subtract rolling 8th-percentile baseline; None to skip
    "baseline_window_s": 60.0,   # window (s) for the rolling baseline
    "normalize_noise": False,    # divide each trace by its robust noise estimate
    "smooth_sigma_s": 0.0,       # Gaussian pre-smoothing (s); 0 to skip
}

# --- output options ---
n_example_neurons = 4            # number of example neurons in the trace panels
example_selection = "active"     # "active" (most stored events) or "random"
window_s = (100.0, 200.0)        # (start, stop) time window shown in the trace panels
save_dir = "/home/rgast/data/parker_data"
seed = 0

# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_session() -> dict:
    """Locate and load the requested session's .mat file."""
    session_dir = f"{path}/{drug}/{dose}"
    match = None
    for folder in os.listdir(session_dir):
        if not os.path.isdir(f"{session_dir}/{folder}"):
            continue
        _, mouse_id, *_ = folder.split("_")
        if mouse_id == mouse or mouse in folder:
            match = folder
            break
    if match is None:
        raise FileNotFoundError(f"No session for mouse '{mouse}' in {session_dir}")
    if condition == "amph":
        match = f"{match}_amph"
    mat = loadmat(f"{session_dir}/{match}/{condition}_drug.mat", simplify_cells=True)
    inner = mat[f"{condition}_drug"]
    traces = np.asarray(inner[input_field], dtype=float)   # (neurons, time)
    events = np.asarray(inner[events_field], dtype=float)  # (neurons, time)
    print(f"Loaded {match}/{condition}_drug.mat: "
          f"{traces.shape[0]} neurons x {traces.shape[1]} frames ({traces.shape[1]/fs:.0f} s)")
    return {"session": match, "traces": traces, "events": events}

# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------

def robust_noise(y: np.ndarray) -> float:
    """Noise estimate robust to transients: MAD of the frame-to-frame difference."""
    dy = np.diff(y)
    return 1.4826 * np.median(np.abs(dy - np.median(dy))) / np.sqrt(2.0) + 1e-12

def preprocess(traces: np.ndarray) -> np.ndarray:
    out = traces.astype(float).copy()

    # rolling-percentile baseline subtraction (detrending)
    pct = preprocessing["detrend_percentile"]
    if pct is not None:
        w = max(1, int(preprocessing["baseline_window_s"] * fs))
        for i in range(out.shape[0]):
            out[i] = out[i] - _rolling_percentile(out[i], w, pct)

    # light Gaussian pre-smoothing
    sig = preprocessing["smooth_sigma_s"] * fs
    if sig > 0:
        out = np.asarray([gaussian_filter1d(out[i], sigma=sig) for i in range(out.shape[0])])

    # noise normalization
    if preprocessing["normalize_noise"]:
        out = np.asarray([out[i] / robust_noise(out[i]) for i in range(out.shape[0])])

    return out

def _rolling_percentile(y: np.ndarray, window: int, pct: float) -> np.ndarray:
    """Rolling-percentile baseline via strided windows (edge-padded)."""
    pad = window // 2
    yp = np.pad(y, pad, mode="edge")
    base = np.empty_like(y)
    for t in range(len(y)):
        base[t] = np.percentile(yp[t:t + window], pct)
    return base

# ---------------------------------------------------------------------------
# OASIS deconvolution (returns the full, non-thresholded activity `s`)
# ---------------------------------------------------------------------------

def oasis_deconvolve(traces: np.ndarray, penalty: int) -> np.ndarray:
    """Per-neuron OASIS deconvolution; returns the continuous activity `s`."""
    from oasis.functions import deconvolve
    g_init = (None,) if ar_order == 1 else (None, None)
    s_all = np.zeros_like(traces, dtype=float)
    for i in range(traces.shape[0]):
        try:
            _, s, _, _, _ = deconvolve(np.asarray(traces[i], dtype=float),
                                       g=g_init, penalty=penalty)
        except Exception:
            continue
        if np.all(np.isfinite(s)):
            s_all[i] = s
    return s_all

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(seed)
    data = load_session()
    stored_events = data["events"]
    proc = preprocess(data["traces"])

    # run OASIS for each penalty
    s_by_penalty = {}
    for label, pen in penalties.items():
        print(f"Running OASIS deconvolution (penalty {label}) ...")
        s_by_penalty[label] = oasis_deconvolve(proc, pen)

    # choose example neurons
    n_neurons = proc.shape[0]
    if example_selection == "active":
        example = np.argsort(stored_events.sum(axis=1))[::-1][:n_example_neurons]
    else:
        example = rng.choice(n_neurons, size=min(n_example_neurons, n_neurons), replace=False)

    # ---- plotting ----
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import seaborn as sb
    plt.rcParams["font.size"] = 12.0
    sb.set_palette("colorblind")
    pal = sb.color_palette("colorblind")
    pen_colors = {"L1": pal[0], "L0": pal[1]}   # OASIS traces
    input_color = "0.7"
    event_color = "black"
    event_alpha = 0.55

    n_ex = len(example)
    fig, axes = plt.subplots(nrows=n_ex, ncols=1, figsize=(15, 2.6 * n_ex),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]

    i0, i1 = int(window_s[0] * fs), int(window_s[1] * fs)
    t = np.arange(i0, i1) / fs

    for row, neuron in enumerate(example):
        ax = axes[row]           # left axis: input calcium trace
        ax2 = ax.twinx()         # right axis: OASIS activity s

        # input calcium trace
        l_in, = ax.plot(t, proc[neuron, i0:i1], color=input_color, lw=1.2, zorder=1,
                        label="input trace")

        # stored spike events as vertical reference lines
        se = i0 + np.where(stored_events[neuron, i0:i1] > 0)[0]
        l_ev = ax.vlines(se / fs, *ax.get_ylim(), color=event_color, lw=1.0, alpha=event_alpha,
                         zorder=0, label="stored spikes")

        # full OASIS output for each penalty
        pen_lines = []
        for label in penalties:
            l, = ax2.plot(t, s_by_penalty[label][neuron, i0:i1], color=pen_colors[label],
                          lw=1.4, alpha=0.9, label=f"OASIS {label}")
            pen_lines.append(l)

        ax.set_ylabel(f"neuron {neuron}\n{input_field.split('_')[0]}")
        ax2.set_ylabel("OASIS activity $s$")
        if row == 0:
            ax.legend(handles=[l_in, l_ev, *pen_lines], ncol=4, loc="lower center",
                      bbox_to_anchor=(0.5, 1.0), fontsize=10)
        if row == n_ex - 1:
            ax.set_xlabel("time (s)")

    fig.suptitle(f"{drug} / {dose} / {mouse} / {condition}  (input: {input_field})", y=1.0)
    fig.tight_layout()
    out = f"{save_dir}/oasis_l0_l1_comparison_{drug}_{mouse}_{condition}.svg"
    fig.canvas.draw()
    plt.savefig(out, format="svg")
    print(f"\nSaved figure to {out}")
    plt.show()

if __name__ == "__main__":
    main()
