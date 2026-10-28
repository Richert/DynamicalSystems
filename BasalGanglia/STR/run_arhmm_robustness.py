"""Robustness checks of the AR-HMM x SPN analysis w.r.t. the neural rate estimate.

Reuses the FIXED K=4 behavioural AR-HMM (arhmm_alldrugs/K4/arhmm_model.pkl), so
the states are identical and only the SPN rates differ:
  norm  -- OASIS L1 rates divided per neuron by the median event amplitude
           (removes brightness differences between neurons), in events/s
  L0    -- OASIS with L0 penalty (discrete events) instead of L1; requires
           `spikes/bin/python extract_oasis_rates.py 0` first
  logstats -- no change of the rates: the current readouts (mean/SD of log-rates +
           third readout) on the fixed K=4 model; outputs of the skewness version
           are in K4_logstats, of the active-neuron correlation version in K4_corrstats
  corrstats -- same as logstats (third readout = mean pairwise correlation of the
           active neurons)
(The dimensionality variants corr / top15 / minmax / minmax_top15 were run with an
earlier version of the readouts; their outputs remain in arhmm_alldrugs/K4_<name>.)

Run one variant per process (they can run in parallel):
  /home/rgast/conda/envs/arhmm/bin/python run_arhmm_robustness.py norm
  /home/rgast/conda/envs/arhmm/bin/python run_arhmm_robustness.py L0
"""

import os
import sys
import shutil
import analysis_arhmm_spn_states as A

base_dir = f"{A.save_dir}/arhmm_alldrugs/K4"

variants = {
    # old floor 1e-6 (raw units) / median event amplitude across all neurons
    # (1.95e-3) x fs = 0.0026 events/s (~0.05 events per 20 s window); the old
    # upper limit was never reached, so no upper clipping here
    "norm": dict(rate_norm="median_event", rate_lims=(2.6e-3, 1e3),
                 h5_path=A.h5_path),
    "L0": dict(rate_norm=None, rate_lims=A.rate_lims,
               h5_path=f"{A.save_dir}/oasis_rates_L0.h5"),
    "logstats": dict(),
    "corrstats": dict(),
}

def main(name):
    out = f"{base_dir}_{name}"
    os.makedirs(out, exist_ok=True)
    shutil.copy(f"{base_dir}/arhmm_model.pkl", f"{out}/arhmm_model.pkl")
    for k, v in variants[name].items():
        setattr(A, k, v)
    A.out_dir, A.load_model = out, True
    print(f"robustness variant '{name}': {variants[name]} -> {out}")
    A.main()


if __name__ == "__main__":
    main(sys.argv[1])
