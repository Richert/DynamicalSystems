"""Raw miniscope movie -> motion correction -> CNMF-E -> OASIS spikes, evaluated
against the original (old) Parker-lab pipeline for the same recording.

Pipeline (CaImAn):
  1. convert the raw MATLAB-written HDF5 movie (dataset "1", 20 Hz) to a local TIFF
  2. motion correction: NoRMCorre (rigid or piecewise-rigid) on high-pass filtered
     frames (standard for 1-photon data), saved as a C-order memmap
  3. cell extraction: CNMF-E (ring background model, corr/PNR seeding) + component
     evaluation (SNR, spatial consistency)
  4. spike inference: OASIS deconvolution inside CNMF-E (AR(1)); a frame holds a
     spike if the deconvolved activity exceeds spike_thr x the neuron's noise level

Evaluation against the old pipeline (events_5hz = "expert" spike annotations at 5 Hz,
cnmfe_traces_5hz, cellXYcoords_um; no spatial footprints are available):
  - new traces are binned to 5 Hz; a global time lag is estimated
  - neurons are matched by trace correlation, constrained by position after fitting
    a similarity transform (scale + translation + axis swap/flip) from new pixel
    centroids to old um centroids; one-to-one via the Hungarian algorithm
  - per matched neuron: loss = mean distance from each old event to the closest new
    spike (s); recall = share of old events with a new spike within +/- tol_s
    (one-to-one); extra = unmatched new spikes per minute

Every step caches its output in work_dir (set the rerun flags to recompute).

Run with the parker env (has CaImAn):
  /home/rgast/conda/envs/parker/bin/python pipeline_raw_spikes.py [<drug>/<dose>/<session> [recording]]
"""

import os
import sys
import glob
import numpy as np
import h5py

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

# paths (environment variables override the defaults, e.g. on the server)
raw_root = os.environ.get("RAW_ROOT", "/mnt/kennedy_labdata/Parkerlab/raw_miniscope")
old_root = os.environ.get("OLD_ROOT", "/mnt/kennedy_labdata/Parkerlab/neural_data")
work_root = os.environ.get("WORK_ROOT", "/home/rgast/data/parker_data/raw_pipeline")

# recording to process: raw folder <drug>/<dose>/<session> and the index of the
# recording in it (files sorted by time): 0 = first, 15-min drug/vehicle recording
# (old session = folder name without "_amph", "veh"), 1 = 45-min amphetamine
# recording (old session = folder name, "amph")
raw_session = "Clozapine/HighDose/20191207_m085_clo04_amph"
recording = 0
# joint mode: both recordings of a day (15-min drug/vehicle + 45-min amphetamine, same
# field of view) are motion-corrected to one template and extracted together by one
# CNMF-E run, so cells are tracked across them; spikes are inferred and evaluated per
# recording. Outputs go to <session>_joint, per-recording files carry _rec0 / _rec1.
joint = False
out_suffix = ""              # appended to per-recording output names (joint mode)
# joint mode, per-recording refit: the joint fit's background and AR decay are dominated
# by the longer amphetamine recording, which degraded the shorter rec0 traces (m797:
# events of the same cells 97 -> 54, recall vs old 0.90 -> 0.78). With refit, only the
# footprints (cell identities) are shared: ring background, traces and AR(1) coefficient
# are re-estimated per recording (refit_iters alternations; outputs carry "_refit")
joint_refit = True
refit_iters = 2

def selection_cli(argv):
    """Component-selection flags (--filt2, --min_snr X, --min_eta R), shared with
    compare_cell_extraction.py: sets them and out_suffix, returns the remaining argv."""
    global filt2, min_snr, min_eta, min_eta_margin, out_suffix
    argv = list(argv)
    if "--filt2" in argv:
        filt2, out_suffix = True, out_suffix + "_f2"
        argv.remove("--filt2")
    for flag in ("--min_snr", "--min_eta", "--min_eta_margin"):
        if flag in argv:
            i = argv.index(flag)
            val = float(argv[i + 1])
            del argv[i:i + 2]
            if flag == "--min_snr":
                min_snr, out_suffix = val, out_suffix + f"_snr{val:g}"
            elif flag == "--min_eta":
                min_eta, out_suffix = val, out_suffix + f"_eta{val:g}"
            else:
                min_eta_margin, out_suffix = val, out_suffix + f"_etam{val:g}"
    return argv

def eta_needed():
    return min_snr is not None or min_eta is not None or min_eta_margin is not None

def apply_cli(argv):
    """CLI: pipeline_raw_spikes.py <raw_session> [recording [min_corr min_pnr]] [--joint]
    (only applied when run as a script, so other scripts can import this module)."""
    global raw_session, recording, joint
    if "--joint" in argv:
        joint = True
        argv = [a for a in argv if a != "--joint"]
    argv = selection_cli(argv)
    if len(argv) > 1:
        raw_session = argv[1]
        recording = int(argv[2]) if len(argv) > 2 else 0
        if len(argv) > 4:
            cnmf_params.update(min_corr=float(argv[3]), min_pnr=float(argv[4]))

fs = 20.0                    # raw frame rate (Hz)
fs_old = 5.0                 # frame rate of the old pipeline's outputs (Hz)

# which steps to recompute (otherwise cached results in work_dir are reused)
rerun = dict(convert=False, motion=False, cnmf=False)
n_processes = int(os.environ.get("N_PROCESSES", 8))   # CaImAn worker processes per run

# motion correction (NoRMCorre)
mc_params = dict(
    pw_rigid=False,          # rigid is usually sufficient for 1p endoscope data
    gSig_filt=(3, 3),        # high-pass filter size used for registration only
    max_shifts=(10, 10),
    strides=(48, 48), overlaps=(24, 24), max_deviation_rigid=3,
    border_nan="copy",
)

# CNMF-E (sizes in pixels of the 192 x 192 movie)
gSig = (3, 3)                # expected half-size of a neuron
cnmf_params = dict(
    method_init="corr_pnr", K=None, gSig=gSig, gSiz=(4 * gSig[0] + 1, 4 * gSig[1] + 1),
    # seeding thresholds: grid on 4 clozapine recordings (min_corr 0.7/0.8 x min_pnr
    # 6/8/10) -> 0.7/6 recovers the most active old neurons (82% vs 73% for 0.8/10)
    # with unchanged match quality, recall, timing and no noise false positives
    center_psf=True, min_corr=0.7, min_pnr=6,
    # AR(1): on the development recording AR(2) gave the same recall at equal event
    # rates but twice the timing loss, noise-driven detections and hardly less
    # rise-edge fragmentation (2-3 frame intervals 11% vs 13%)
    p=1, merge_thr=0.7, rf=40, stride=20, tsub=2, ssub=1,
    nb=0, nb_patch=0, low_rank_background=None, ssub_B=2, ring_size_factor=1.4,
    only_init=True, normalize_init=False, update_background_components=True,
    del_duplicates=True, method_deconvolution="oasis", decay_time=1.0,
)
quality = dict(min_SNR=3.0, rval_thr=0.85, use_cnn=False)
# footprint shape filter (after CaImAn's quality check), on the footprint above 30% of
# its max: rejects lens-edge arcs, vessel/neuropil bands and fragmented components.
# On 4 clozapine recordings matched (real) cells have area ~50 px (95%: <=112 px at
# 3.3 um/px), elongation ~1.4, solidity ~0.96; these limits reject ~1% of them
shape_filter = dict(max_area_px=200, max_elongation=3.5, min_solidity=0.75, single_part=True)
# stricter post-filter (CLI flag --filt2, outputs carry "_f2"): the overlay figures showed
# implausible large components (lens edge, bright background) and duplicates in dense
# fields of view. Area relative to the recording's median cell area (cell size in pixels
# differs between mice, median 41-72 px; a fixed 120 px cut removed 16% of m106's cells):
# on 4 recordings, > 2 x median holds 3% of matched (real) and 16% of unmatched cells.
# Duplicates: of two accepted cells whose cores (> 50% of max) overlap by more than
# max_core_overlap (of the smaller core), the one with the lower SNR is dropped.
# (A core-vs-ring contrast test on the correlation image separated matched and unmatched
# cells too poorly to be used: a cut removing 29% of unmatched also removed 12% of matched.)
filt2 = False
post_filter = dict(max_area_rel=2.0, max_core_overlap=0.5)
# looser quality check (CLI --min_snr X, outputs carry "_snrX"): accepted cells pass
# CaImAn's check by SNR alone (r_values ~0.3, rval_thr 0.85 is practically never met),
# and at seeding min_corr 0.6 many compact, bright correlation-image blobs that look
# like cells are rejected with SNR 2-3 (m106/m120/m797: 40/56/102 vs 91/104/209
# accepted). With min_snr, components are re-selected from the stored SNR_comp/r_values
# (no CNMF-E rerun).
min_snr = None
# event-triggered footprint consistency (CLI --min_eta R, outputs carry "_etaR"): at the
# component's strongest transients (peaks of C), the frame minus the mean of the frames
# 0.5-1 s earlier (removes static structure such as the bright lens edge), averaged and
# correlated with the footprint in a box around it. Real cells: a blob matching the
# footprint; lens-edge / background components whose 'activity' is not visible in the
# dF/F frames: low correlation. (Peak dF/F did not separate them: dim real cells have
# the lowest dF/F.) Computed for all components with SNR >= eta_params["min_snr"],
# cached in eta_<tag>.csv. On 5 recordings (SNR >= 2) r >= 0.4 looked best for m971/m040,
# r >= 0.3 for m797/m120: a fixed r is not equally strict across fields of view.
# Margin (CLI --min_eta_margin M, outputs carry "_etamM"): r minus the largest r of the
# same event-triggered average with the footprint shifted by shift_px in 4 directions
# (same box size around the shifted position). A real cell matches its own footprint
# better than displaced copies; co-active neighbours in dense fields of view raise both,
# smeared lens-edge / background fluctuations fit displaced copies about as well.
# eta: DataFrame indexed by component (eta_r, eta_margin, n_events); None: not used
min_eta = None
min_eta_margin = None
eta_params = dict(n_events=20, min_sep_s=1.0, base_window_s=(1.0, 0.5), box_half_px=12, shift_px=8,
                  min_snr=2.0)
eta = None
# extraction diagnostics added to the evaluation summary (set by accepted_components)
extra_summary = {}

def cnmf_tag():
    """Identifies a CNMF-E setting in cache/output names (AR order + seeding thresholds)."""
    return f"p{cnmf_params['p']}_c{cnmf_params['min_corr']:g}_pnr{cnmf_params['min_pnr']:g}"

# spike detection on the OASIS output
# spike if deconvolved activity > spike_thr x noise level; 1.5-2 gives high recall of
# the old events and SPN-like rates (~0.1-0.2 Hz) with no noise-driven detections
# (dev recording: 1.5 -> recall 0.87, 0.21 Hz; 2.0 -> recall 0.85, 0.12 Hz; 2.0 is the
# more conservative alternative, to revisit if downstream readouts call for it)
spike_thr = 1.5
fig_thresholds = [1.5]       # one full evaluation + figure per threshold (e.g. [1.5, 2.0])
thr_sweep = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]   # thresholds for the sweep
# burst spikes: a neighbouring spike of the same neuron within [min, max) s; intervals
# below 0.2 s (<= 3 frames) are mostly one rising edge split by the deconvolution
burst_isi_s = (0.2, 0.5)

# evaluation
max_lag_frames = 20          # search +/- this many raw frames for the old/new time lag
r_seed = 0.7                 # trace correlation of mutual-best pairs used to fit the geometry
r_min = 0.4                  # min trace correlation of an accepted old/new neuron pair
max_dist_um = 20.0           # max centroid distance (after the transform) of a pair
tol_s = 0.25                 # spike-matching tolerance (s); old bins are 0.2 s wide
min_old_events = 5           # reference = old cells with >= this many events (clear activity)
n_examples = 4               # example neurons in the figure

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def session_paths(rec=None):
    """Raw file, old-pipeline file, condition and work dir of recording `rec`
    (default: the global `recording`; joint mode: shared <session>_joint dir)."""
    rec = recording if rec is None else rec
    drug_raw, dose, folder = raw_session.split("/")
    files = sorted(glob.glob(f"{raw_root}/{raw_session}/*.h5"),
                   key=lambda f: os.path.basename(f).split("_video")[0][-19:])
    raw_file = files[rec]
    # old pipeline: drug folder names are lower-case except abbreviations
    drug_old = next(d for d in os.listdir(old_root) if d.lower() == drug_raw.lower())
    if rec == 0:
        old_session, cond = folder.replace("_amph", ""), "veh"
    else:
        old_session, cond = folder, "amph"
    old_file = f"{old_root}/{drug_old}/{dose}/{old_session}/{cond}_drug.mat"
    work = f"{work_root}/{folder}_joint" if joint else f"{work_root}/{folder}_rec{rec}"
    os.makedirs(work, exist_ok=True)
    return raw_file, old_file, cond, work

# ---------------------------------------------------------------------------
# 1-4: preprocessing, extraction, spikes
# ---------------------------------------------------------------------------

def convert(raw_file, work, name="raw.tif"):
    """Raw HDF5 movie (dataset "1", T x H x W uint16) -> local TIFF. Returns (path, frames)."""
    import tifffile
    out = f"{work}/{name}"
    if os.path.isfile(out) and not rerun["convert"]:
        with tifffile.TiffFile(out) as t:
            return out, len(t.pages)
    with h5py.File(raw_file, "r") as h:
        mov = h["1"][...]                         # ~1.3 GB (15 min) / ~4 GB (45 min)
    tifffile.imwrite(out, mov, bigtiff=True)
    print(f"converted {raw_file} -> {out} {mov.shape} {mov.dtype}")
    return out, mov.shape[0]

def motion_correct(tifs, work, dview):
    """NoRMCorre on one or several movies (joint mode: both recordings of a day, one
    template); returns a single C-order memmap (movies concatenated in time)."""
    import caiman as cm
    from caiman.motion_correction import MotionCorrect
    done = glob.glob(f"{work}/memmap_*_order_C_*.mmap")
    if done and not rerun["motion"]:
        # joint mode leaves per-movie memmaps next to the concatenated one: take the longest
        return max(done, key=lambda f: int(f.split("_frames_")[1].split(".")[0]))
    mc = MotionCorrect(list(tifs), dview=dview, **mc_params)
    mc.motion_correct(save_movie=True)
    fname_mc = mc.fname_tot_els if mc_params["pw_rigid"] else mc.fname_tot_rig
    shifts = np.array(mc.shifts_rig) if not mc_params["pw_rigid"] else None
    if shifts is not None:
        np.save(f"{work}/mc_shifts.npy", shifts)
        print(f"motion correction: max |shift| = {np.abs(shifts).max():.1f} px")
    fname = cm.save_memmap(fname_mc, base_name="memmap_", order="C", border_to_0=0)
    for f in fname_mc:                            # drop the intermediate F-order movie
        os.remove(f)
    # CaImAn writes memmaps to its temp dir (set to work_dir in main); make sure the
    # result stays with this recording (the file name encodes the dimensions)
    dest = f"{work}/{os.path.basename(fname)}"
    if os.path.abspath(fname) != os.path.abspath(dest):
        import shutil
        shutil.move(fname, dest)
    return dest

def run_cnmfe(fname, work, dview):
    import caiman as cm
    from caiman.source_extraction.cnmf import cnmf, params
    out = f"{work}/cnmfe_{cnmf_tag()}.hdf5"            # setting in the name: no stale reuse
    Yr, dims, T = cm.load_memmap(fname)
    images = Yr.T.reshape((T,) + dims, order="F")
    if os.path.isfile(out) and not rerun["cnmf"]:
        cnm = cnmf.load_CNMF(out)
        cn = np.load(f"{work}/corr_image.npy")
        return cnm, cn, dims
    step = max(1, T // 4000)                     # summary images on a subsample
    cn, pnr = cm.summary_images.correlation_pnr(images[::step], gSig=gSig[0], swap_dim=False)
    np.save(f"{work}/corr_image.npy", cn)
    np.save(f"{work}/pnr_image.npy", pnr)
    opts = params.CNMFParams(params_dict=dict(fr=fs, **cnmf_params))
    cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, params=opts)
    cnm.fit(images)
    cnm.params.set("quality", quality)
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
    print(f"CNMF-E: {cnm.estimates.A.shape[1]} components, "
          f"{len(cnm.estimates.idx_components)} accepted")
    cnm.save(out)
    return cnm, cn, dims

def footprint_shape(a, frac=0.3):
    """(area px, elongation, solidity, n_parts) of a 2-D footprint above frac x max."""
    from skimage.measure import label, regionprops
    lab = label(a > frac * a.max())
    p = max(regionprops(lab), key=lambda r: r.area)
    return p.area, p.major_axis_length / max(p.minor_axis_length, 1e-6), p.solidity, lab.max()

def quality_passed(est):
    """Components passing CaImAn's quality check, or with min_snr the looser re-selection
    (SNR >= min_snr or r_value >= rval_thr) from the stored values."""
    if min_snr is None:
        return [int(k) for k in est.idx_components]
    snr, rval = np.asarray(est.SNR_comp).ravel(), np.asarray(est.r_values).ravel()
    return [int(k) for k in np.flatnonzero((snr >= min_snr) | (rval >= quality["rval_thr"]))]

def compute_eta(cnm, fname, work, dims):
    """Event-triggered footprint consistency (see eta_params) of all components with
    SNR >= eta_params["min_snr"]: r and its margin over shifted footprints; cached in
    eta_<tag>.csv. Returns a DataFrame indexed by component."""
    import pandas as pd
    out = f"{work}/eta_{cnmf_tag()}.csv"
    if os.path.isfile(out):
        d = pd.read_csv(out)
        if "eta_margin" in d:                     # (older caches: r only -> recompute)
            return d.set_index("component")
    import caiman as cm
    from scipy.signal import find_peaks
    est = cnm.estimates
    comps = np.flatnonzero(np.asarray(est.SNR_comp).ravel() >= eta_params["min_snr"])
    Yr, _, T = cm.load_memmap(fname)
    Y = np.array(Yr, dtype=np.float32)            # pixels (F order) x frames, into RAM once
    b0, b1 = (int(round(s * fs)) for s in eta_params["base_window_s"])
    h, s = eta_params["box_half_px"], eta_params["shift_px"]

    def box_r(img, fp, y, x):                     # correlation in the box around (y, x)
        ys, xs = slice(max(0, y - h), min(dims[0], y + h + 1)), slice(max(0, x - h), min(dims[1], x + h + 1))
        return np.corrcoef(img[ys, xs].ravel(), fp[ys, xs].ravel())[0, 1]

    def shifted(a, dy, dx):                       # footprint moved by (dy, dx), zero fill
        out = np.zeros_like(a)
        out[max(0, dy):dims[0] + min(0, dy), max(0, dx):dims[1] + min(0, dx)] = \
            a[max(0, -dy):dims[0] + min(0, -dy), max(0, -dx):dims[1] + min(0, -dx)]
        return out

    rows = []
    for k in comps:
        a = est.A[:, k].toarray().reshape(dims, order="F")
        y, x = np.unravel_index(np.argmax(a), dims)
        c = np.asarray(est.C[k]).ravel()
        pk, _ = find_peaks(c, height=0, distance=max(1, int(eta_params["min_sep_s"] * fs)))
        pk = pk[pk >= b0]
        pk = pk[np.argsort(c[pk])[::-1][:eta_params["n_events"]]]
        if len(pk) == 0:
            rows.append(dict(component=int(k), eta_r=np.nan, eta_margin=np.nan, n_events=0))
            continue
        img = np.mean([Y[:, t] - Y[:, t - b0:t - b1].mean(1) for t in pk], 0).reshape(dims, order="F")
        r = box_r(img, a, y, x)
        r_sh = [box_r(img, shifted(a, dy, dx), y + dy, x + dx)
                for dy, dx in [(s, 0), (-s, 0), (0, s), (0, -s)]
                if 0 <= y + dy < dims[0] and 0 <= x + dx < dims[1]]
        rows.append(dict(component=int(k), eta_r=r, eta_margin=r - np.nanmax(r_sh), n_events=len(pk)))
    del Y
    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    print(f"event-triggered footprint consistency of {len(d)} components: median r {d.eta_r.median():.2f}, "
          f"margin {d.eta_margin.median():.2f}")
    return d.set_index("component")

def accepted_components(cnm, dims, cn=None):
    """Indices of components passing CaImAn's quality check (or the min_snr re-selection)
    AND the shape filter (and, with min_eta, the event-triggered footprint consistency;
    with filt2, the stricter post-filter). With the correlation image cn, also counts
    its peaks (>= 0.6) not covered by any accepted cell (-> extra_summary)."""
    est = cnm.estimates
    passed = quality_passed(est)
    keep, rejected, areas = [], [], {}
    for k in passed:
        a = est.A[:, k].toarray().reshape(dims, order="F")
        area, el, sol, parts = footprint_shape(a)
        bad = (area > shape_filter["max_area_px"] or el > shape_filter["max_elongation"]
               or sol < shape_filter["min_solidity"]
               or (shape_filter["single_part"] and parts > 1))
        (rejected if bad else keep).append(int(k))
        areas[int(k)] = area
    print(f"shape filter: {len(keep)} of {len(passed)} accepted components kept")
    for col, lim in [("eta_r", min_eta), ("eta_margin", min_eta_margin)]:
        if lim is None or not keep:
            continue
        val = eta[col].reindex(keep).values                    # not computed (nan): kept
        low = [k for k, v in zip(keep, val) if v < lim]
        rejected += low
        keep = [k for k in keep if k not in set(low)]
        extra_summary[f"n_{col}_dropped"] = len(low)
        print(f"event-triggered footprint consistency ({col} < {lim:g}): {len(low)} dropped, {len(keep)} kept")
    if filt2 and keep:
        lim = post_filter["max_area_rel"] * np.median([areas[k] for k in keep])
        big = [k for k in keep if areas[k] > lim]
        rejected += big
        keep = [k for k in keep if areas[k] <= lim]
        print(f"relative area filter (> {lim:.0f} px): {len(big)} dropped, {len(keep)} kept")
    cores = {k: (lambda a: a > 0.5 * a.max())(est.A[:, k].toarray().ravel()) for k in keep}
    if filt2 and keep:
        snr = np.asarray(est.SNR_comp).ravel()
        drop = set()
        for i, k in enumerate(keep):
            for l in keep[i + 1:]:
                if k in drop or l in drop:
                    continue
                inter = np.sum(cores[k] & cores[l])
                if inter > post_filter["max_core_overlap"] * min(cores[k].sum(), cores[l].sum()):
                    drop.add(k if snr[k] < snr[l] else l)
        rejected += sorted(drop)
        keep = [k for k in keep if k not in drop]
        print(f"duplicate filter: {len(drop)} overlapping components dropped, {len(keep)} kept")
    if cn is not None:
        from skimage.feature import peak_local_max
        peaks = peak_local_max(cn, min_distance=4, threshold_abs=0.6, exclude_border=4)
        cover = np.zeros(dims, bool)
        for k in keep:
            a = est.A[:, k].toarray().reshape(dims, order="F")
            cover |= a > 0.3 * a.max()
        unc = sum(not cover[y, x] for y, x in peaks)
        extra_summary.update(n_corr_peaks=len(peaks), n_corr_peaks_uncovered=int(unc))
        print(f"correlation-image peaks >= 0.6: {len(peaks)}, {unc} without an accepted cell")
    return np.array(keep, dtype=int), rejected

def refit_segment(cnm, fname, seg, dims, work, name):
    """Per-recording refit with the joint footprints fixed (all components, so rejected
    ones do not leak into the background): alternate (1) ring background W, b0 fitted
    on this recording only (CaImAn's compute_W, as in CNMF-E) and (2) raw traces
    Yraw = C + YrA (projection of the background-corrected movie on the footprints,
    neighbour overlap removed with the current C), denoised by OASIS with an AR(1)
    coefficient estimated from this recording. Returns C, Yraw, g for all components."""
    import caiman as cm
    from caiman.source_extraction.cnmf.initialization import compute_W
    from caiman.source_extraction.cnmf.estimates import Estimates
    from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi
    out = f"{work}/refit_{cnmf_tag()}_{name}.npz"
    if os.path.isfile(out) and not rerun["cnmf"]:
        d = np.load(out)
        return d["C"], d["Yraw"], d["g"]
    est = cnm.estimates
    A = est.A.tocsc()
    Yr, _, _ = cm.load_memmap(fname)
    Y = np.asarray(Yr[:, seg], dtype=np.float32)               # pixels x frames
    C = np.asarray(est.C[:, seg], dtype=np.float32).copy()
    T = Y.shape[1]
    nA2 = np.ravel(A.power(2).sum(0)) + np.finfo(np.float32).eps
    AA = (A.T @ A).toarray()
    radius = cnmf_params["ring_size_factor"] * cnmf_params["gSiz"][0]
    g = np.zeros(len(C))
    for it in range(refit_iters):
        W, b0 = compute_W(Y, A.toarray(), C, dims, radius, ssub=cnmf_params["ssub_B"])
        bg = Estimates(A=A, C=C, b=None, f=None, dims=dims)
        bg.W, bg.b0 = W, b0
        AtY = np.zeros_like(C)
        for t0 in range(0, T, 2000):                           # background in frame chunks
            t1 = min(T, t0 + 2000)
            bg.C = C[:, t0:t1]
            AtY[:, t0:t1] = A.T @ (Y[:, t0:t1] - bg.compute_background(Y[:, t0:t1]))
        Yraw = C + (AtY - AA @ C) / nA2[:, None]
        for k in range(len(C)):
            c, _, _, gk, _, _, _ = constrained_foopsi(Yraw[k], p=cnmf_params["p"],
                                                      method_deconvolution="oasis")
            C[k], g[k] = c, np.atleast_1d(gk)[0]
        print(f"refit {name} iteration {it + 1}: median g {np.median(g):.4f}")
    Yraw = C + (AtY - AA @ C) / nA2[:, None]                   # residuals w.r.t. the final C
    np.savez_compressed(out, C=C, Yraw=Yraw, g=g)
    return C, Yraw, g

def deconvolve_all(cnm, dims, segments=None, refit=None, cn=None):
    """Accepted components: denoised traces C, OASIS deconvolution S of the raw trace
    (C + residual YrA) and S_noise of a noise-only surrogate (the sign-flipped
    residual: realistic noise, no calcium transients), both with the neuron's own
    AR coefficient and noise level, so real and surrogate use the identical
    detector. Also returns the raw extracted traces Yraw = C + YrA, the component
    indices, noise levels sn and centroids (row, col) in pixels.
    segments: list of frame slices (joint mode: one per recording) deconvolved
    separately, each with its own noise level; sn is then (cells, segments).
    refit: per segment (C, Yraw, g) of all components from refit_segment, used in place
    of the joint C, YrA and AR coefficient."""
    import caiman as cm
    from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi, GetSn
    est = cnm.estimates
    idx, _ = accepted_components(cnm, dims, cn)
    C, YrA = est.C[idx].copy(), est.YrA[idx].copy()
    segs = segments or [slice(0, C.shape[1])]
    gs = [np.asarray(est.g)[idx]] * len(segs)
    if refit is not None:
        for seg, (Cs, Ys, g_s), s in zip(segs, refit, range(len(segs))):
            C[:, seg], YrA[:, seg] = Cs[idx], Ys[idx] - Cs[idx]
            gs[s] = g_s[idx]
    S, S_noise = np.zeros_like(C), np.zeros_like(C)
    p = cnmf_params["p"]
    sn = np.zeros((len(idx), len(segs)))
    for k in range(len(idx)):
        for s, seg in enumerate(segs):
            y = C[k, seg] + YrA[k, seg]
            sn[k, s] = np.asarray(est.neurons_sn)[idx][k] if segments is None else GetSn(y)
            for out, trace in [(S, y), (S_noise, -YrA[k, seg])]:
                *_, sp, _ = constrained_foopsi(trace, g=np.atleast_1d(gs[s][k])[:p], sn=sn[k, s],
                                               p=p, method_deconvolution="oasis")
                out[k, seg] = sp
    if segments is None:
        sn = sn[:, 0]
    r = np.mean([np.corrcoef(S[k], est.S[idx][k])[0, 1] for k in range(len(idx))
                 if S[k].std() > 0 and est.S[idx][k].std() > 0])
    print(f"re-deconvolved {len(idx)} traces (mean corr with CNMF-E's S: {r:.3f})")
    cents = cm.base.rois.com(est.A[:, idx], *dims)
    return C, C + YrA, S, S_noise, sn, cents, idx

def detect(S, sn, thr):
    """Binary event trains: deconvolved activity above thr x the neuron's noise level,
    with runs of consecutive above-threshold frames merged into one event at the run
    onset (OASIS spreads one calcium-transient onset over adjacent 50 ms frames, so
    adjacent frames are not separate spikes; separate increments within a transient
    remain separate events -> bursts)."""
    above = S > thr * sn[:, None]
    onset = above.copy()
    onset[:, 1:] &= ~above[:, :-1]
    return onset

# ---------------------------------------------------------------------------
# evaluation against the old pipeline
# ---------------------------------------------------------------------------

def load_old(old_file, cond):
    from scipy.io import loadmat
    d = loadmat(old_file, simplify_cells=True)[f"{cond}_drug"]
    return (np.asarray(d["events_5hz"], float) > 0, np.asarray(d["cnmfe_traces_5hz"], float),
            np.asarray(d["cellXYcoords_um"], float))

def to_old_rate(X, lag):
    """Bin a (n, T) 20 Hz array to 5 Hz after shifting by `lag` raw frames
    (new frame f corresponds to old time (f - lag) / fs)."""
    k = int(fs / fs_old)
    X = X[:, lag:] if lag >= 0 else np.pad(X, ((0, 0), (-lag, 0)))
    n = X.shape[1] // k
    return X[:, :n * k].reshape(X.shape[0], n, k).mean(axis=2)

def corr_matrix(A, B):
    """Pearson correlation between rows of A and rows of B (same length)."""
    def z(X):
        X = X - X.mean(1, keepdims=True)
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return z(A) @ z(B).T

def estimate_lag(C_new, C_old):
    """Raw-frame lag maximising the mean of each old neuron's best correlation."""
    best = (-np.inf, 0)
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        Cn = to_old_rate(C_new, lag)
        T = min(Cn.shape[1], C_old.shape[1])
        score = np.mean(np.sort(corr_matrix(C_old[:, :T], Cn[:, :T]).max(1))[-20:])
        if score > best[0]:
            best = (score, lag)
    return best[1]

AXES = [(sw, fx, fy) for sw in (False, True) for fx in (1, -1) for fy in (1, -1)]

def fit_transform(p_new, p_old):
    """Similarity transform (axis swap/flip, isotropic scale, translation) mapping new
    pixel centroids (row, col) onto old um centroids (x, y). Returns (fn, rms)."""
    best = None
    for sw, fx, fy in AXES:
        q = p_new[:, ::-1] if sw else p_new.copy()
        q = q * np.array([fx, fy])
        X = np.column_stack([q.ravel(), np.tile([1, 0], len(q)), np.tile([0, 1], len(q))])
        sol, *_ = np.linalg.lstsq(X, p_old.ravel(), rcond=None)
        s, tx, ty = sol
        if s <= 0:
            continue
        rms = np.sqrt(np.mean(np.sum((s * q + [tx, ty] - p_old) ** 2, axis=1)))
        if best is None or rms < best[0]:
            best = (rms, sw, fx, fy, s, tx, ty)
    rms, sw, fx, fy, s, tx, ty = best
    def fn(p):
        q = p[:, ::-1] if sw else p
        return s * q * np.array([fx, fy]) + [tx, ty]
    print(f"geometry: swap={sw} flip=({fx},{fy}) scale={s:.2f} um/px  rms={rms:.1f} um")
    return fn, rms, s

def match_neurons(R, cents_new, xy_old):
    """One-to-one matching on trace correlation R (old x new), constrained by the
    fitted geometry. Returns (pairs [(i_old, j_new)], transform)."""
    from scipy.optimize import linear_sum_assignment
    bo, bn = R.argmax(1), R.argmax(0)
    seed = [(i, j) for i, j in enumerate(bo) if bn[j] == i and R[i, j] >= r_seed]
    fn = None
    if len(seed) >= 5:
        fn, rms, s = fit_transform(cents_new[[j for _, j in seed]], xy_old[[i for i, _ in seed]])
    else:
        print(f"only {len(seed)} seed pairs -> matching on correlation only")
    cost = 1.0 - R
    if fn is not None:
        D = np.linalg.norm(xy_old[:, None, :] - fn(cents_new)[None, :, :], axis=2)
        cost[D > max_dist_um] = 1e6
    cost[R < r_min] = 1e6
    rows, cols = linear_sum_assignment(cost)
    pairs = [(i, j) for i, j in zip(rows, cols) if cost[i, j] < 1e6]
    return pairs, fn

def spike_metrics(t_old, t_new, duration_s):
    """loss = mean |old event - closest new spike| (s); recall within tol_s with
    one-to-one greedy matching (closest first); extra = unmatched new spikes/min."""
    if len(t_old) == 0:
        return dict(n_old=0, n_new=len(t_new), loss=np.nan, recall=np.nan,
                    extra_per_min=len(t_new) / duration_s * 60)
    if len(t_new) == 0:
        return dict(n_old=len(t_old), n_new=0, loss=np.inf, recall=0.0, extra_per_min=0.0)
    d = np.abs(t_old[:, None] - t_new[None, :])
    loss = float(d.min(1).mean())
    used_o, used_n, hits = set(), set(), 0
    for flat in np.argsort(d, axis=None):
        i, j = np.unravel_index(flat, d.shape)
        if d[i, j] > tol_s:
            break
        if i in used_o or j in used_n:
            continue
        used_o.add(i); used_n.add(j); hits += 1
    return dict(n_old=len(t_old), n_new=len(t_new), loss=loss, recall=hits / len(t_old),
                extra_per_min=(len(t_new) - hits) / duration_s * 60)

def new_times(spk_row, lag, T):
    """Spike times (s, old time axis) of one new neuron within the common window."""
    f = np.flatnonzero(spk_row) - lag
    return (f[(f >= 0) & (f < T * fs / fs_old)] + 0.5) / fs

def pair_metrics(pairs, R, spk, E_old, lag, T):
    from pandas import DataFrame
    duration = T / fs_old
    rows = []
    for i, j in pairs:
        t_old = (np.flatnonzero(E_old[i]) + 0.5) / fs_old
        rows.append(dict(old=i, new=j, r=R[i, j],
                         **spike_metrics(t_old, new_times(spk[j], lag, T), duration)))
    return DataFrame(rows)

def burst_fraction(times_list):
    """Share of spikes with a neighbouring spike of the same neuron at an interval in
    burst_isi_s = [min, max), and all inter-spike intervals (s)."""
    n_burst = n = 0
    isis = []
    lo, hi = burst_isi_s
    for t in times_list:
        if len(t) < 2:
            n += len(t)
            continue
        d = np.diff(t)
        isis.append(d)
        b = (d >= lo) & (d < hi)
        near = np.zeros(len(t), bool)
        near[:-1] |= b
        near[1:] |= b
        n_burst += near.sum(); n += len(t)
    return n_burst / max(n, 1), (np.concatenate(isis) if isis else np.array([]))

def active_old(E_old):
    """Old cells used as reference: those with clear activity (>= min_old_events)."""
    return np.flatnonzero(E_old.sum(1) >= min_old_events)

def evaluate(C, S, S_noise, sn, cents, E_old, C_old, xy_old):
    # reference = old cells with clear activity; silent/near-silent old cells are
    # excluded before matching (they cannot be matched by trace correlation anyway
    # and must not claim new cells). old_index maps back to the old pipeline's rows.
    n_old_total = len(E_old)
    old_index = active_old(E_old)
    E_old, C_old, xy_old = E_old[old_index], C_old[old_index], xy_old[old_index]
    print(f"reference: {len(old_index)} of {n_old_total} old cells with >= {min_old_events} events")
    lag = estimate_lag(C, C_old)
    C5 = to_old_rate(C, lag)
    T = min(C5.shape[1], C_old.shape[1])
    C5, C_old, E_old = C5[:, :T], C_old[:, :T], E_old[:, :T]
    print(f"time lag: {lag} raw frames ({lag / fs:.2f} s); common length {T} old frames")
    R = corr_matrix(C_old, C5)
    pairs, fn = match_neurons(R, cents, xy_old)      # independent of the spike threshold
    minutes = T / fs_old / 60

    # threshold sweep: recall on matched neurons, rates and noise false positives on all
    sweep = []
    for thr in thr_sweep:
        spk, fp = detect(S, sn, thr), detect(S_noise, sn, thr)
        d = pair_metrics(pairs, R, spk, E_old, lag, T)
        rate = np.array([len(new_times(s, lag, T)) for s in spk]) / minutes
        fp_rate = np.array([len(new_times(s, lag, T)) for s in fp]) / minutes
        sweep.append(dict(thr=thr, recall=d.recall.mean(), loss_median=np.median(d.loss[np.isfinite(d.loss)]),
                          rate_median=np.median(rate), fp_rate_median=np.median(fp_rate),
                          fp_share=fp_rate.sum() / max(rate.sum(), 1e-12),
                          burst_frac=burst_fraction([new_times(s, lag, T) for s in spk])[0]))
    from pandas import DataFrame
    sweep = DataFrame(sweep)

    spk, fp = detect(S, sn, spike_thr), detect(S_noise, sn, spike_thr)
    df = pair_metrics(pairs, R, spk, E_old, lag, T)
    df["fp_per_min"] = [len(new_times(fp[j], lag, T)) / minutes for j in df.new]
    df.insert(1, "old_index", old_index[df.old.values] if len(df) else [])
    t_new = [new_times(s, lag, T) for s in spk]
    t_old = [(np.flatnonzero(e) + 0.5) / fs_old for e in E_old]
    bf_new, isi_new = burst_fraction(t_new)
    bf_old, isi_old = burst_fraction(t_old)
    rate_new = np.array([len(t) for t in t_new]) / minutes
    fp_new = np.array([len(new_times(s, lag, T)) for s in fp]) / minutes
    summary = dict(n_old=len(C_old), n_old_total=n_old_total, n_new=len(C),
                   n_matched=len(pairs), lag_frames=lag,
                   spike_thr=spike_thr,
                   loss_median=float(np.median(df.loss[np.isfinite(df.loss)])) if len(df) else np.nan,
                   recall_mean=float(df.recall.mean()) if len(df) else np.nan,
                   extra_per_min_mean=float(df.extra_per_min.mean()) if len(df) else np.nan,
                   rate_new_median_hz=float(np.median(rate_new) / 60),
                   rate_old_median_hz=float(np.median([len(t) for t in t_old]) / minutes / 60),
                   noise_fp_share=float(fp_new.sum() / max(rate_new.sum(), 1e-12)),
                   min_corr=cnmf_params["min_corr"], min_pnr=cnmf_params["min_pnr"],
                   # cell-detection scores: share of the (active) old reference cells
                   # recovered, match quality, new cells without an old counterpart
                   frac_active_old_matched=float(len(pairs) / max(len(C_old), 1)),
                   n_old_active=int(len(C_old)),
                   r_matched_median=float(np.median([R[i, j] for i, j in pairs])) if pairs else np.nan,
                   n_new_unmatched=int(len(C) - len(pairs)),
                   burst_frac_new=float(bf_new), burst_frac_old=float(bf_old))
    return df, summary, dict(lag=lag, T=T, R=R, pairs=pairs, fn=fn, C5=C5, C_old=C_old,
                             E_old=E_old, xy_old=xy_old, old_index=old_index,
                             spk=spk, sweep=sweep, isi_new=isi_new,
                             isi_old=isi_old, rate_new=rate_new, fp_new=fp_new)

# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def make_figure(work, cn, cents, spk, C, Yraw, ev, df, summary):
    xy_old = ev["xy_old"]                                   # reference (active) old cells
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    lag, T, pairs, fn = ev["lag"], ev["T"], ev["pairs"], ev["fn"]
    E_old, C_old = ev["E_old"], ev["C_old"]
    old_c, new_c = "#3b6fb6", "#d1495b"
    fig = plt.figure(figsize=(16, 19), layout="constrained")
    gs = GridSpec(4, 3, figure=fig, height_ratios=[1.2, 1, 1, 1.6])

    # (1) neuron positions, in old um coordinates (new centroids transformed)
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(*xy_old.T, s=30, facecolors="none", edgecolors=old_c, label=f"old (n={len(xy_old)})")
    if fn is not None:
        pn = fn(cents)
        ax.scatter(*pn.T, s=10, color=new_c, label=f"new (n={len(cents)})")
        for i, j in pairs:
            ax.plot([xy_old[i, 0], pn[j, 0]], [xy_old[i, 1], pn[j, 1]], c="0.4", lw=0.8)
    ax.set(title=f"neuron positions ({len(pairs)} matched)", xlabel="x (um)", ylabel="y (um)",
           aspect="equal")
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="lower right")

    # (2) population rate over time (events/s per neuron, 10 s smoothing)
    ax = fig.add_subplot(gs[0, 1:])
    k = int(10 * fs_old)
    spk5 = to_old_rate(spk.astype(float), lag)[:, :T] * (fs / fs_old)   # spikes per old bin
    t = np.arange(T) / fs_old / 60
    for X, c, lab in [(E_old, old_c, "old events"), (spk5, new_c, "new spikes")]:
        pr = X.sum(0) / len(X) * fs_old
        ax.plot(t, np.convolve(pr, np.ones(k) / k, mode="same"), c=c, label=lab)
    ax.set(title="population rate (10 s smoothing)", xlabel="time (min)",
           ylabel="events / s / neuron")
    ax.legend(fontsize=9)

    # (3) per-neuron rate distribution over the session
    ax = fig.add_subplot(gs[1, 0])
    dur = T / fs_old / 60
    r_old, r_new = E_old.sum(1) / dur, spk5.sum(1) / dur
    bins = np.logspace(-1, np.log10(max(r_old.max(), r_new.max(), 1) * 1.2), 25)
    ax.hist(np.clip(r_old, 0.1, None), bins=bins, alpha=0.6, color=old_c,
            label=f"old (median {np.median(r_old):.1f}/min)")
    ax.hist(np.clip(r_new, 0.1, None), bins=bins, alpha=0.6, color=new_c,
            label=f"new (median {np.median(r_new):.1f}/min)")
    ax.set(xscale="log", title="per-neuron event rate", xlabel="events / min (0 -> 0.1)",
           ylabel="# neurons")
    ax.legend(fontsize=8)

    # (4) per-neuron loss
    ax = fig.add_subplot(gs[1, 1])
    fin = df.loss[np.isfinite(df.loss)]
    ax.hist(fin, bins=25, color="0.4")
    ax.axvline(np.median(fin), c="k", ls="--")
    ax.set(title=f"per-neuron loss (median {np.median(fin):.2f} s; "
                 f"{int((~np.isfinite(df.loss)).sum())} without new spikes)",
           xlabel="mean |old event - closest new spike| (s)", ylabel="# matched neurons")

    # (4b) recall vs extra spikes
    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(df.extra_per_min, df.recall, c=df.r, cmap="viridis", s=25)
    ax.set(title=f"recall (+/-{tol_s} s) vs extra spikes; mean recall "
                 f"{summary['recall_mean']:.2f}", xlabel="extra new spikes / min",
           ylabel="recall of old events")

    # (6) threshold sweep: recall vs rate vs noise false positives
    sw = ev["sweep"]
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(sw.thr, sw.recall, "o-", c="k", label="recall of old events")
    ax.plot(sw.thr, sw.fp_share, "s-", c="0.55", label="noise FP / real spikes")
    ax.plot(sw.thr, sw.burst_frac, "^-", c=new_c,
            label=f"burst share (ISI {burst_isi_s[0]}-{burst_isi_s[1]} s)")
    ax.axvline(spike_thr, c="k", ls=":", lw=1)
    ax.set(ylim=(0, 1.05), xlabel="spike threshold (x noise level)", ylabel="fraction",
           title="threshold sweep")
    ax2 = ax.twinx()
    ax2.plot(sw.thr, sw.rate_median / 60, "d--", c=new_c, alpha=0.6)
    ax2.axhline(np.median(E_old.sum(1)) / (T / fs_old), c=old_c, ls="--", lw=1)
    ax2.set_ylabel("median rate (Hz); dashed blue = old", color=new_c)
    ax.legend(fontsize=7, loc="center right")

    # (7) inter-spike intervals, old vs new
    ax = fig.add_subplot(gs[2, 1])
    bins = np.logspace(np.log10(0.04), np.log10(600), 40)
    for isi, c, lab, bf in [(ev["isi_old"], old_c, "old", summary["burst_frac_old"]),
                            (ev["isi_new"], new_c, "new", summary["burst_frac_new"])]:
        ax.hist(isi, bins=bins, density=True, histtype="step", lw=1.5, color=c,
                label=f"{lab} (burst share {bf:.2f})")
    for b in burst_isi_s:
        ax.axvline(b, c="k", ls=":", lw=1)
    ax.set(xscale="log", xlabel="inter-spike interval (s)", ylabel="density",
           title="inter-spike intervals (old resolution 0.2 s)")
    ax.legend(fontsize=8)

    # (8) per-neuron real spike rate vs noise false-positive rate
    ax = fig.add_subplot(gs[2, 2])
    ax.scatter(ev["rate_new"] / 60, ev["fp_new"] / 60, s=10, c=new_c, alpha=0.6)
    lim = max(ev["rate_new"].max() / 60, 1e-3) * 1.1
    ax.plot([0, lim], [0, lim], c="0.6", lw=1)
    ax.set(xlabel="spike rate (Hz)", ylabel="noise false-positive rate (Hz)",
           title=f"noise FP share {summary['noise_fp_share']:.2f} at thr={spike_thr}")

    # (5) example traces: matched neurons with the most old events
    ex = df.sort_values("n_old", ascending=False).head(n_examples)
    sub = gs[3, :].subgridspec(len(ex), 1)
    win = slice(0, int(180 * fs_old))                       # first 3 min
    t5 = np.arange(T)[win] / fs_old
    for a, (_, row) in enumerate(ex.iterrows()):
        ax = fig.add_subplot(sub[a])
        i, j = int(row.old), int(row.new)
        zo = C_old[i, win] / (C_old[i, win].max() + 1e-12)
        cn_den = C[j, max(lag, 0):]
        tn = np.arange(len(cn_den)) / fs
        keep = tn < t5[-1]
        scale = cn_den[keep].max() + 1e-12          # raw and denoised share one scale
        ax.plot(tn[keep], Yraw[j, max(lag, 0):][keep] / scale, c="0.7", lw=0.5)
        ax.plot(t5, zo + 1.2, c=old_c, lw=1)
        ax.plot(tn[keep], cn_den[keep] / scale, c=new_c, lw=0.8)
        ax.plot(t5[E_old[i, win]], np.full(E_old[i, win].sum(), 2.35), "|", c=old_c, ms=8)
        fs_new = (np.flatnonzero(spk[j]) - lag) / fs
        fs_new = fs_new[(fs_new >= 0) & (fs_new < t5[-1])]
        ax.plot(fs_new, np.full(len(fs_new), -0.15), "|", c=new_c, ms=8)
        ax.set(yticks=[], ylabel=f"old {int(row.old_index)}\nnew {j}", xlim=(0, t5[-1]))
        ax.text(1.0, 1.0, f"r={row.r:.2f}  loss={row.loss:.2f}s  recall={row.recall:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)
        if a < len(ex) - 1:
            ax.set_xticklabels([])
    ax.set_xlabel("time (s); old CNMF-E trace (blue, top) + events; new CNMF-E denoised "
                  "trace (red) on the raw extracted trace C + residual (grey) + spikes")

    fig.suptitle(f"{raw_session} rec{recording}: new pipeline (CaImAn CNMF-E + OASIS) vs old "
                 f"| old {summary['n_old']} active (of {summary['n_old_total']}) / new "
                 f"{summary['n_new']} / matched {summary['n_matched']} neurons", fontsize=13)
    out = f"{work}/pipeline_vs_old_{cnmf_tag()}_thr{spike_thr}{out_suffix}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"saved figure {out}")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def save_and_evaluate(work, cn, cents, idx, C, Yraw, S, S_noise, sn, old_file, cond):
    """Save one recording's outputs and evaluate it against the old pipeline (file
    names carry out_suffix, e.g. _rec0 / _rec1 in joint mode)."""
    import json
    global spike_thr                                 # looped over fig_thresholds below
    sfx = out_suffix
    np.savez_compressed(f"{work}/new_pipeline_outputs_{cnmf_tag()}{sfx}.npz", C=C, Yraw=Yraw,
                        S=S, S_noise=S_noise, sn=sn, centroids_px=cents, component=idx, fs=fs)
    np.save(f"{work}/spikes_{cnmf_tag()}_thr{spike_thr}{sfx}.npy", detect(S, sn, spike_thr))
    if not os.path.isfile(old_file):                 # e.g. folder-name mismatch (m85 vs m085)
        print(f"no old-pipeline file {old_file}: extraction saved, evaluation skipped")
        return
    E_old, C_old, xy_old = load_old(old_file, cond)
    for spike_thr in fig_thresholds:          # full evaluation + figure per threshold
        tag = f"{cnmf_tag()}_thr{spike_thr}"
        df, summary, ev = evaluate(C, S, S_noise, sn, cents, E_old, C_old, xy_old)
        df.insert(3, "component", idx[df.new.values] if len(df) else [])   # CNMF-E component id
        df.to_csv(f"{work}/matched_neurons_{tag}{sfx}.csv", index=False)
        summary.update(extra_summary, filt2=filt2, min_snr=min_snr or quality["min_SNR"], min_eta=min_eta,
                       min_eta_margin=min_eta_margin)
        json.dump(summary, open(f"{work}/summary_{tag}{sfx}.json", "w"), indent=1)
        np.save(f"{work}/spikes_{tag}{sfx}.npy", ev["spk"])
        print(f"\n[{tag}{sfx}] summary:", summary)
        make_figure(work, cn, cents, ev["spk"], C, Yraw, ev, df, summary)
    ev["sweep"].to_csv(f"{work}/threshold_sweep_{cnmf_tag()}{sfx}.csv", index=False)
    print(ev["sweep"].round(3).to_string(index=False))

def main():
    import caiman as cm
    global recording, out_suffix
    recs = [0, 1] if joint else [recording]
    paths = [session_paths(r) for r in recs]          # (raw, old, cond, work) per recording
    work = paths[0][3]
    # CaImAn's temp files (memmaps) are named only by movie dimensions/frame count,
    # so parallel runs must not share a temp dir: use this recording's work dir
    os.environ["CAIMAN_TEMP"] = work
    for (raw_file, old_file, _, _), r in zip(paths, recs):
        print(f"rec{r} raw: {raw_file}\nrec{r} old: {old_file}")
    print(f"work: {work}")
    _, dview, _ = cm.cluster.setup_cluster(backend="multiprocessing", n_processes=n_processes)
    try:
        tifs = [convert(p[0], work, "raw.tif" if not joint else f"raw_rec{r}.tif")
                for p, r in zip(paths, recs)]
        fname = motion_correct([t for t, _ in tifs], work, dview)
        cnm, cn, dims = run_cnmfe(fname, work, dview)
    finally:
        cm.stop_server(dview=dview)
    global eta
    if eta_needed():                                  # metric also stored for evaluation
        eta = compute_eta(cnm, fname, work, dims)
    if not joint:
        C, Yraw, S, S_noise, sn, cents, idx = deconvolve_all(cnm, dims, cn=cn)
        save_and_evaluate(work, cn, cents, idx, C, Yraw, S, S_noise, sn, *paths[0][1:3])
        return
    # joint: one set of cells; spikes inferred and evaluated per recording segment
    bounds = np.cumsum([0] + [n for _, n in tifs])
    segs = [slice(bounds[i], bounds[i + 1]) for i in range(len(tifs))]
    print(f"joint extraction: segments {[(s.start, s.stop) for s in segs]}")
    refit = ([refit_segment(cnm, fname, seg, dims, work, f"rec{r}") for r, seg in zip(recs, segs)]
             if joint_refit else None)
    C, Yraw, S, S_noise, sn, cents, idx = deconvolve_all(cnm, dims, segments=segs, refit=refit, cn=cn)
    for i, (r, seg) in enumerate(zip(recs, segs)):
        recording, out_suffix = r, f"_rec{r}" + ("_refit" if joint_refit else "")
        save_and_evaluate(work, cn, cents, idx, C[:, seg], Yraw[:, seg], S[:, seg],
                          S_noise[:, seg], sn[:, i], *paths[i][1:3])


if __name__ == "__main__":
    apply_cli(sys.argv)
    main()
