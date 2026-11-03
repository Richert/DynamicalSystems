#!/bin/bash
# CNMF-E settings test on one recording (m85, 15-min recording): each setting against
# CaImAn's default, one at a time, plus all four non-merging changes combined at each
# merging threshold. Selection and spike detection as in the final batch. Default run =
# the existing outputs (snr2_eta0.4_sz_rim5). CNMF-E results are cached per setting
# (cnmfe_<tag>.hdf5, tag incl. the non-default settings; refit: ..._refit).
work_dir="$HOME/PycharmProjects/DynamicalSystems/BasalGanglia/STR"
data_root="/media/storage/DATA"
export RAW_ROOT="$data_root/Parkerlab/raw_miniscope"
export OLD_ROOT="$data_root/Parkerlab/neural_data"
export WORK_ROOT="$data_root/Parkerlab/results/raw_pipeline"
session="Clozapine/Vehicle/20191205_m85_clo01_amph"
rec=0
corr=0.6
pnr=5
sel="--min_snr 2 --min_eta 0.4 --size_rel --lens_rim 5 --detect_sum 2"
log_dir="$WORK_ROOT/logs_cnmfe_settings_test"
mkdir -p $log_dir
cd $work_dir
export PATH="$HOME/miniforge3/envs/parker/bin:$PATH"
export N_PROCESSES=${N_PROCESSES:-6} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
all4="--refit --nrgthr 0.99 --ring 1.8 --init_iter 3"
variants=(
  "refit|--refit"
  "nrg0.99|--nrgthr 0.99"
  "ring1.8|--ring 1.8"
  "it3|--init_iter 3"
  "m0.6|--merge 0.6"
  "m0.8|--merge 0.8"
  "all4_m0.7|$all4"
  "all4_m0.6|$all4 --merge 0.6"
  "all4_m0.8|$all4 --merge 0.8"
)
for v in "${variants[@]}"; do
  name="${v%%|*}"
  flags="${v#*|}"
  ( python pipeline_raw_spikes.py "$session" $rec $corr $pnr $sel $flags > "$log_dir/$name.log" 2>&1
    echo "Finished $name (exit $?)" ) &
done
wait
echo "All CNMF-E settings runs finished."
grep -H "refit:\|CNMF-E:\|shape filter\|consistency\|relative size\|lens rim" $log_dir/*.log
