#!/bin/bash

# Run pipeline_raw_spikes.py (default settings) on every raw recording of one drug,
# in parallel batches: both recordings of each session (0 = 15-min drug/vehicle,
# 1 = 45-min amphetamine). Ends with a summary table of the evaluation vs the old
# pipeline. Recordings without an old-pipeline counterpart are extracted only.

# paths on the server (/media/storage/DATA there = /mnt/kennedy_labdata on the workstation)
work_dir="$HOME/PycharmProjects/DynamicalSystems/BasalGanglia/STR"
data_root="/media/storage/DATA"
export RAW_ROOT="$data_root/Parkerlab/raw_miniscope"
export OLD_ROOT="$data_root/Parkerlab/neural_data"
# results go to the shared drive (= /mnt/kennedy_labdata/Parkerlab/results/... on the workstation)
export WORK_ROOT="$data_root/Parkerlab/results/raw_pipeline"
log_dir="$WORK_ROOT/logs"

# environment (needs CaImAn: mamba install -n parker -c conda-forge caiman)
cd $work_dir
source $HOME/.bashrc
mamba activate parker
mkdir -p $log_dir

# set condition
drug="Clozapine"
recs=( 0 1 )
# parallel runs per recording type: the grid run peaked at ~75 GB for 5 parallel
# 15-min recordings (~15 GB each); 45-min recordings have 3x the frames (~45 GB)
batch_sizes=( 10 4 )           # rec 0 (15 min): ~150 GB, rec 1 (45 min): ~180 GB of 250 GB

# limit threads per run: parallelism comes from CaImAn's worker processes
export N_PROCESSES=8
n_threads=1
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads

run() {   # session (<drug>/<dose>/<folder>), recording index
  name="$(basename $1)_rec$2"
  echo "Starting $name"
  python pipeline_raw_spikes.py "$1" "$2" > "$log_dir/$name.log" 2>&1
  echo "Finished $name (exit $?)"
}

# all sessions of the drug, both recordings each (all 15-min ones first, then 45-min)
for rec in "${recs[@]}"; do
  batch_size=${batch_sizes[$rec]}
  for dir in "$RAW_ROOT/$drug"/*/*/; do
    session="${dir#$RAW_ROOT/}"
    session="${session%/}"
    run "$session" "$rec" &
    if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
      wait -n
    fi
  done
  wait                         # finish this recording type before the next (memory)
done

echo "All jobs finished."
grep -L "summary" $log_dir/*.log | sed 's/^/no evaluation summary: /'
python collect_raw_pipeline_grid.py "$WORK_ROOT" 1.5
