#!/bin/bash
# CNMF-E settings test, round 2 (m85, 15-min recording): footprint trimming 0.99 + background
# ring 1.8 (2 seeding rounds, no refit) at merging thresholds 0.6 / 0.7 / 0.8. Per fit two
# selections: the final one (consistency r >= 0.4) and a loose one without the consistency
# filter (SNR >= 2, size + lens rim), whose match table covers every candidate, for sweeping
# the SNR / consistency thresholds locally. The second selection reuses the cached fit.
work_dir="$HOME/PycharmProjects/DynamicalSystems/BasalGanglia/STR"
data_root="/media/storage/DATA"
export RAW_ROOT="$data_root/Parkerlab/raw_miniscope"
export OLD_ROOT="$data_root/Parkerlab/neural_data"
export WORK_ROOT="$data_root/Parkerlab/results/raw_pipeline"
session="Clozapine/Vehicle/20191205_m85_clo01_amph"
rec=0
corr=0.6
pnr=5
final="--min_snr 2 --min_eta 0.4 --size_rel --lens_rim 5 --detect_sum 2"
loose="--min_snr 2 --size_rel --lens_rim 5 --detect_sum 2"
log_dir="$WORK_ROOT/logs_cnmfe_settings_test2"
mkdir -p $log_dir
cd $work_dir
export PATH="$HOME/miniforge3/envs/parker/bin:$PATH"
export N_PROCESSES=${N_PROCESSES:-8} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for m in 0.6 0.7 0.8; do
  flags="--nrgthr 0.99 --ring 1.8 --merge $m"
  ( python pipeline_raw_spikes.py "$session" $rec $corr $pnr $final $flags > "$log_dir/m${m}_final.log" 2>&1
    python pipeline_raw_spikes.py "$session" $rec $corr $pnr $loose $flags > "$log_dir/m${m}_loose.log" 2>&1
    echo "Finished merge $m (exit $?)" ) &
done
# default fit, loose selection (match table for the baseline)
( python pipeline_raw_spikes.py "$session" $rec $corr $pnr $loose > "$log_dir/default_loose.log" 2>&1
  echo "Finished default loose (exit $?)" ) &
wait
echo "All round-2 runs finished."
