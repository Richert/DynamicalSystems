#!/bin/bash

# Overlay figures (compare_cell_extraction.py) of the final full run only (the run's own overlay
# step failed: correlation image not passed to the final rules); reads the existing fits.
# paths on the server (/media/storage/DATA there = /mnt/kennedy_labdata on the workstation)
work_dir="$HOME/PycharmProjects/DynamicalSystems/BasalGanglia/STR"
data_root="/media/storage/DATA"
export RAW_ROOT="$data_root/Parkerlab/raw_miniscope"
export OLD_ROOT="$data_root/Parkerlab/neural_data"
# results go to the shared drive (= /mnt/kennedy_labdata/Parkerlab/results/... on the workstation)
export WORK_ROOT="$data_root/Parkerlab/results/raw_pipeline"

# extraction settings (seeding thresholds, component selection flags)
corr=0.6
pnr=5
sel="--nrgthr 0.999 --min_snr 2 --size_rel --lens_rim 5 --hybrid 0.3 0.2 --vessel 3 --offpeak 6 --frag_edge 1.05 --detect_sum 1.5"
sfx="final_nrg0.999"
spike_tag="1.5_sum3"               # summary_<...>_thr1.5_sum3 (collect_raw_pipeline_grid.py)
log_dir="$WORK_ROOT/logs_$sfx"
export FIG_DIR="$WORK_ROOT/cell_overlays_$sfx"

# environment (needs CaImAn: mamba install -n parker -c conda-forge caiman)
cd $work_dir
source $HOME/.bashrc
mamba activate parker 2>/dev/null
export PATH="$HOME/miniforge3/envs/parker/bin:$PATH"   # (mamba activate fails in non-interactive shells)
mkdir -p $log_dir $FIG_DIR
drug="Clozapine"
batch_size=6                   # overlays load the motion-corrected movie (up to ~15 GB each)
export OMP_NUM_THREADS=2

for rec in 0 1; do
  for dir in "$RAW_ROOT/$drug"/*/*/; do
    session="${dir#$RAW_ROOT/}"
    session="${session%/}"
    name="$(basename $session)_rec$rec"
    ( python compare_cell_extraction.py "$session" "$rec" "$WORK_ROOT" $corr $pnr $sel \
        > "$log_dir/${name}_overlay.log" 2>&1; echo "Finished $name (overlay exit $?)" ) &
    if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
      wait -n
    fi
  done
done
wait
echo "$(ls $FIG_DIR/*_cell_overlay_*.png | wc -l) overlay figures in $FIG_DIR"
