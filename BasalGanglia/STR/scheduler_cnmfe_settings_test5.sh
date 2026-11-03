#!/bin/bash
# CNMF-E settings test, round 5: the five test recordings (15-min recordings of m971, m040,
# m106, m120, m797) with trimming 0.999 and ring 1.6 / 1.8 / 2.0 x merging 0.65 / 0.7 / 0.75
# (45 fits), plus merging 0.65 at the three ring sizes on m85 (rings x 0.7 / 0.75 done in
# round 4). Only the loose selection (SNR >= 2, size + lens rim, no consistency filter): its
# consistency table and match table cover every candidate, so the consistency threshold and the
# hybrid / vessel / fragment / off-peak rules are swept locally afterwards.
# Jobs run through a queue of MAX_JOBS parallel fits (N_PROCESSES cores each).
work_dir="$HOME/PycharmProjects/DynamicalSystems/BasalGanglia/STR"
data_root="/media/storage/DATA"
export RAW_ROOT="$data_root/Parkerlab/raw_miniscope"
export OLD_ROOT="$data_root/Parkerlab/neural_data"
export WORK_ROOT="$data_root/Parkerlab/results/raw_pipeline"
corr=0.6
pnr=5
loose="--min_snr 2 --size_rel --lens_rim 5 --detect_sum 2 --nrgthr 0.999"
log_dir="$WORK_ROOT/logs_cnmfe_settings_test5"
mkdir -p $log_dir
cd $work_dir
export PATH="$HOME/miniforge3/envs/parker/bin:$PATH"
export N_PROCESSES=${N_PROCESSES:-8} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
MAX_JOBS=${MAX_JOBS:-8}
sessions=(Clozapine/Vehicle/20191205_m971_clo01_amph Clozapine/Vehicle/20200110_m040_clo01_amph
          Clozapine/Vehicle/20200224_m106_clo01_amph Clozapine/HighDose/20200228_m120_clo04_amph
          Clozapine/HighDose/20210107_m797_clo04_amph)
jobs_list=()
for s in "${sessions[@]}"; do
  for r in 1.6 1.8 2.0; do
    for m in 0.65 0.7 0.75; do
      jobs_list+=("$s|$r|$m")
    done
  done
done
for r in 1.6 1.8 2.0; do
  jobs_list+=("Clozapine/Vehicle/20191205_m85_clo01_amph|$r|0.65")
done
for j in "${jobs_list[@]}"; do
  IFS="|" read -r session r m <<< "$j"
  name="$(basename $session)_ring${r}_m${m}"
  ( python pipeline_raw_spikes.py "$session" 0 $corr $pnr $loose --ring $r --merge $m > "$log_dir/$name.log" 2>&1
    echo "Finished $name (exit $?)" ) &
  while [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; do
    wait -n
  done
done
wait
echo "All round-5 runs finished (${#jobs_list[@]} fits)."
