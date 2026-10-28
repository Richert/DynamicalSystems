#!/bin/bash

# CNMF-E seeding grid (min_corr x min_pnr) of pipeline_raw_spikes.py on a few
# recordings, run in parallel batches. Stage 1 runs one setting per recording (this
# also converts the movie and does the motion correction, which all settings of a
# recording share); stage 2 runs the remaining settings. Ends with a summary table.

# paths on the server (/media/storage/DATA there = /mnt/kennedy_labdata on the workstation)
work_dir="$HOME/PycharmProjects/DynamicalSystems/BasalGanglia/STR"
data_root="/media/storage/DATA"
export RAW_ROOT="$data_root/Parkerlab/raw_miniscope"
export OLD_ROOT="$data_root/Parkerlab/neural_data"
export WORK_ROOT="$HOME/data/parker_data/raw_pipeline"
log_dir="$WORK_ROOT/grid_logs"

# environment (needs CaImAn: mamba install -n parker -c conda-forge caiman)
cd $work_dir
source $HOME/.bashrc
mamba activate parker
mkdir -p $log_dir

# set condition
recordings=( "Clozapine/HighDose/20191207_m085_clo04_amph"
             "Clozapine/Vehicle/20200722_m404_clo01_amph"
             "Clozapine/Vehicle/20191205_m972_clo01_amph"
             "Clozapine/HighDose/20200918_m485_clo04_amph" )
rec_idx=0                      # 0 = 15-min drug/vehicle recording, 1 = 45-min amphetamine recording
corrs=( 0.8 0.7 )
pnrs=( 10 8 6 )
batch_size=5                   # parallel runs (~10-15 GB RAM each for a 15-min recording)

# limit threads per run: parallelism comes from CaImAn's worker processes
export N_PROCESSES=8
n_threads=1
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads

run() {   # recording, min_corr, min_pnr
  name="$(basename $1)_rec${rec_idx}_c$2_pnr$3"
  echo "Starting $name"
  python pipeline_raw_spikes.py "$1" "$rec_idx" "$2" "$3" > "$log_dir/$name.log" 2>&1
  echo "Finished $name (exit $?)"
}

# stage 1: preprocessing + first setting, one run per recording
for rec in "${recordings[@]}"; do
  run "$rec" "${corrs[0]}" "${pnrs[0]}" &
  if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
    wait -n
  fi
done
wait

# stage 2: remaining settings (motion correction is reused from stage 1)
for rec in "${recordings[@]}"; do
  for c in "${corrs[@]}"; do
    for p in "${pnrs[@]}"; do
      if [[ "$c" == "${corrs[0]}" && "$p" == "${pnrs[0]}" ]]; then
        continue
      fi
      run "$rec" "$c" "$p" &
      if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
        wait -n
      fi
    done
  done
done
wait

echo "All jobs finished."
python collect_raw_pipeline_grid.py "$WORK_ROOT" 1.5
