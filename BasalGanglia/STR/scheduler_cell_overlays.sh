#!/bin/bash

# Run compare_cell_extraction.py (old vs new cells on representative frames) for every
# recording of one drug that pipeline_raw_spikes.py has processed, in parallel. Figures
# and diagnostics are collected in one folder (FIG_DIR). Light on memory (the movie is
# read from the memmap on demand), mostly disk-bound.

# paths on the server (/media/storage/DATA there = /mnt/kennedy_labdata on the workstation)
work_dir="$HOME/PycharmProjects/DynamicalSystems/BasalGanglia/STR"
data_root="/media/storage/DATA"
export RAW_ROOT="$data_root/Parkerlab/raw_miniscope"
export OLD_ROOT="$data_root/Parkerlab/neural_data"
export WORK_ROOT="$data_root/Parkerlab/results/raw_pipeline"
export FIG_DIR="$WORK_ROOT/cell_overlays"
log_dir="$FIG_DIR/logs"

cd $work_dir
source $HOME/.bashrc
mamba activate parker
mkdir -p $log_dir

drug="Clozapine"
batch_size=10

n_threads=1
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads

run() {   # session (<drug>/<dose>/<folder>), recording index
  name="$(basename $1)_rec$2"
  echo "Starting $name"
  python compare_cell_extraction.py "$1" "$2" > "$log_dir/$name.log" 2>&1
  echo "Finished $name (exit $?)"
}

for dir in "$RAW_ROOT/$drug"/*/*/; do
  session="${dir#$RAW_ROOT/}"
  session="${session%/}"
  for rec in 0 1; do
    [[ -d "$WORK_ROOT/$(basename $session)_rec$rec" ]] || continue
    run "$session" "$rec" &
    if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
      wait -n
    fi
  done
done
wait
echo "All jobs finished: $(ls $FIG_DIR/*_cell_overlay_*.png | wc -l) figures in $FIG_DIR"
