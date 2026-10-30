#!/bin/bash

# Extraction test after the visual check of the overlay figures (missed dim cells,
# implausible large / duplicate components): lower CNMF-E seeding thresholds
# (min_corr x min_pnr) x default vs stricter post-filter (--filt2: area <= 2 x the recording median,
# duplicates removed) on 5 rec0 recordings (motion correction reused). Then overlay
# figures for every setting and a summary table.

# paths on the server (/media/storage/DATA there = /mnt/kennedy_labdata on the workstation)
work_dir="$HOME/PycharmProjects/DynamicalSystems/BasalGanglia/STR"
data_root="/media/storage/DATA"
export RAW_ROOT="$data_root/Parkerlab/raw_miniscope"
export OLD_ROOT="$data_root/Parkerlab/neural_data"
export WORK_ROOT="$data_root/Parkerlab/results/raw_pipeline"
export FIG_DIR="$WORK_ROOT/extraction_test/overlays"
log_dir="$WORK_ROOT/extraction_test/logs"

cd $work_dir
source $HOME/.bashrc
mamba activate parker
mkdir -p $log_dir $FIG_DIR

# missed cells: m106, m120; implausible components: m971; typical: m040, m797
recordings=( "Clozapine/Vehicle/20200224_m106_clo01_amph"
             "Clozapine/HighDose/20200228_m120_clo04_amph"
             "Clozapine/Vehicle/20191205_m971_clo01_amph"
             "Clozapine/Vehicle/20200110_m040_clo01_amph"
             "Clozapine/HighDose/20210107_m797_clo04_amph" )
rec_idx=0
corrs=( 0.6 0.65 0.7 )
pnrs=( 5 6 )
batch_size=10                  # parallel runs (~15 GB RAM each for a 15-min recording)

export N_PROCESSES=8
n_threads=1
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads

run() {   # recording, min_corr, min_pnr: CNMF-E + default filter, then the stricter filter
  # (the second run reuses the cached CNMF-E model), then overlay figures for both
  name="$(basename $1)_rec${rec_idx}_c$2_pnr$3"
  echo "Starting $name"
  python pipeline_raw_spikes.py "$1" "$rec_idx" "$2" "$3" > "$log_dir/$name.log" 2>&1
  python pipeline_raw_spikes.py "$1" "$rec_idx" "$2" "$3" --filt2 > "$log_dir/${name}_f2.log" 2>&1
  python compare_cell_extraction.py "$1" "$rec_idx" "$WORK_ROOT" "$2" "$3" > "$log_dir/${name}_overlay.log" 2>&1
  python compare_cell_extraction.py "$1" "$rec_idx" "$WORK_ROOT" "$2" "$3" --filt2 > "$log_dir/${name}_overlay_f2.log" 2>&1
  echo "Finished $name"
}

for rec in "${recordings[@]}"; do
  for c in "${corrs[@]}"; do
    for p in "${pnrs[@]}"; do
      run "$rec" "$c" "$p" &
      if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
        wait -n
      fi
    done
  done
done
wait

echo "All jobs finished."
names=$(for r in "${recordings[@]}"; do printf "%s_rec%s," "$(basename $r)" "$rec_idx"; done)
python collect_raw_pipeline_grid.py "$WORK_ROOT" 1.5 "${names%,}" > "$WORK_ROOT/extraction_test/summary.txt"
cp "$WORK_ROOT/seeding_grid_summary_thr1.5.csv" "$WORK_ROOT/extraction_test/"
cat "$WORK_ROOT/extraction_test/summary.txt"
