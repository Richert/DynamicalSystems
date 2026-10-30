#!/bin/bash

# Component-selection test after the extraction test (seeding min_corr 0.6 / min_pnr 5,
# cached CNMF-E fits reused): looser quality check (--min_snr 2.5 / 2, many rejected
# SNR 2-3 components look like cells) and the event-triggered footprint consistency
# filter (--min_eta, against lens-edge / background components whose activity is not
# visible in the dF/F frames), on the 5 rec0 recordings of the extraction test. Overlay
# figures for every setting and a summary table.

# paths on the server (/media/storage/DATA there = /mnt/kennedy_labdata on the workstation)
work_dir="$HOME/PycharmProjects/DynamicalSystems/BasalGanglia/STR"
data_root="/media/storage/DATA"
export RAW_ROOT="$data_root/Parkerlab/raw_miniscope"
export OLD_ROOT="$data_root/Parkerlab/neural_data"
export WORK_ROOT="$data_root/Parkerlab/results/raw_pipeline"
out_dir="$WORK_ROOT/snr_test"
export FIG_DIR="$out_dir/overlays"
log_dir="$out_dir/logs"

cd $work_dir
source $HOME/.bashrc
mamba activate parker
mkdir -p $log_dir $FIG_DIR

recordings=( "Clozapine/Vehicle/20200224_m106_clo01_amph"
             "Clozapine/HighDose/20200228_m120_clo04_amph"
             "Clozapine/Vehicle/20191205_m971_clo01_amph"
             "Clozapine/Vehicle/20200110_m040_clo01_amph"
             "Clozapine/HighDose/20210107_m797_clo04_amph" )
rec_idx=0
corr=0.6
pnr=5
# selection settings (the first one computes and caches the consistency metric)
settings=( "--min_eta 0.4"
           "--min_snr 2.5"
           "--min_snr 2"
           "--min_snr 2 --min_eta 0.3"
           "--min_snr 2 --min_eta 0.4" )

export N_PROCESSES=8
n_threads=1
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads

run() {   # recording: all selection settings in sequence (shared metric cache), each
  # followed by its overlay figure
  base="$(basename $1)_rec${rec_idx}"
  echo "Starting $base"
  for s in "${settings[@]}"; do
    name="${base}$(echo $s | tr -d ' -' | sed 's/min_/_/g')"
    python pipeline_raw_spikes.py "$1" "$rec_idx" "$corr" "$pnr" $s > "$log_dir/$name.log" 2>&1
    python compare_cell_extraction.py "$1" "$rec_idx" "$WORK_ROOT" "$corr" "$pnr" $s > "$log_dir/${name}_overlay.log" 2>&1
  done
  echo "Finished $base"
}

for rec in "${recordings[@]}"; do
  run "$rec" &
done
wait

echo "All jobs finished."
names=$(for r in "${recordings[@]}"; do printf "%s_rec%s," "$(basename $r)" "$rec_idx"; done)
python collect_raw_pipeline_grid.py "$WORK_ROOT" 1.5 "${names%,}" > "$out_dir/summary.txt"
cp "$WORK_ROOT/seeding_grid_summary_thr1.5.csv" "$out_dir/"
cat "$out_dir/summary.txt"
