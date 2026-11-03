#!/bin/bash
# CNMF-E settings test, round 3 (m85, 15-min recording): fine-tuning around trimming 0.99 /
# ring 1.8 / merging 0.8, one setting varied at a time: trimming 0.98 / 0.995, ring 1.6 / 2.0,
# merging 0.75. Per fit the final and the loose selection (no consistency filter, for the
# local hybrid / vessel / fragment-rule evaluation).
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
log_dir="$WORK_ROOT/logs_cnmfe_settings_test3"
mkdir -p $log_dir
cd $work_dir
export PATH="$HOME/miniforge3/envs/parker/bin:$PATH"
export N_PROCESSES=${N_PROCESSES:-8} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
variants=(
  "nrg0.98|--nrgthr 0.98 --ring 1.8 --merge 0.8"
  "nrg0.995|--nrgthr 0.995 --ring 1.8 --merge 0.8"
  "ring1.6|--nrgthr 0.99 --ring 1.6 --merge 0.8"
  "ring2.0|--nrgthr 0.99 --ring 2.0 --merge 0.8"
  "m0.75|--nrgthr 0.99 --ring 1.8 --merge 0.75"
)
for v in "${variants[@]}"; do
  name="${v%%|*}"
  flags="${v#*|}"
  ( python pipeline_raw_spikes.py "$session" $rec $corr $pnr $final $flags > "$log_dir/${name}_final.log" 2>&1
    python pipeline_raw_spikes.py "$session" $rec $corr $pnr $loose $flags > "$log_dir/${name}_loose.log" 2>&1
    echo "Finished $name (exit $?)" ) &
done
wait
echo "All round-3 runs finished."
