#!/bin/bash
# Final CNMF-E settings test (six recordings: 15-min recordings of m971, m040, m106, m120, m797,
# m85): merging 0.7, trimming 0.999 / 0.9999 (CaImAn default) x ring 1.4 / 1.5 / 1.6.
# 0.999 / 1.6 exists from round 5, 0.9999 / 1.4 is the default fit (cached: selection only).
# Loose selection only (SNR >= 2, size + lens rim): the final rules (hybrid, vessel, off-peak,
# fragment / edge piece) are applied locally to these results.
work_dir="$HOME/PycharmProjects/DynamicalSystems/BasalGanglia/STR"
data_root="/media/storage/DATA"
export RAW_ROOT="$data_root/Parkerlab/raw_miniscope"
export OLD_ROOT="$data_root/Parkerlab/neural_data"
export WORK_ROOT="$data_root/Parkerlab/results/raw_pipeline"
corr=0.6
pnr=5
loose="--min_snr 2 --size_rel --lens_rim 5 --detect_sum 2"
log_dir="$WORK_ROOT/logs_cnmfe_settings_final"
mkdir -p $log_dir
cd $work_dir
export PATH="$HOME/miniforge3/envs/parker/bin:$PATH"
export N_PROCESSES=${N_PROCESSES:-8} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
MAX_JOBS=${MAX_JOBS:-8}
sessions=(Clozapine/Vehicle/20191205_m971_clo01_amph Clozapine/Vehicle/20200110_m040_clo01_amph
          Clozapine/Vehicle/20200224_m106_clo01_amph Clozapine/HighDose/20200228_m120_clo04_amph
          Clozapine/HighDose/20210107_m797_clo04_amph Clozapine/Vehicle/20191205_m85_clo01_amph)
settings=("0.999|1.4" "0.999|1.5" "0.9999|1.5" "0.9999|1.6" "0.9999|1.4")   # last: default fit
for s in "${sessions[@]}"; do
  for st in "${settings[@]}"; do
    IFS="|" read -r nrg ring <<< "$st"
    name="$(basename $s)_nrg${nrg}_ring${ring}"
    ( python pipeline_raw_spikes.py "$s" 0 $corr $pnr $loose --nrgthr $nrg --ring $ring > "$log_dir/$name.log" 2>&1
      echo "Finished $name (exit $?)" ) &
    while [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; do
      wait -n
    done
  done
done
wait
echo "All final-test runs finished."
