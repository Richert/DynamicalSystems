#!/bin/bash
# Re-evaluate the four m85 recordings against the old pipeline (raw folders "m85", old
# folders "m085": fixed in session_paths); everything up to CNMF-E and the consistency
# tables is cached, so this reruns selection, spike inference, evaluation and overlays.
work_dir="$HOME/PycharmProjects/DynamicalSystems/BasalGanglia/STR"
data_root="/media/storage/DATA"
export RAW_ROOT="$data_root/Parkerlab/raw_miniscope"
export OLD_ROOT="$data_root/Parkerlab/neural_data"
export WORK_ROOT="$data_root/Parkerlab/results/raw_pipeline"
corr=0.6
pnr=5
sel="--min_snr 2 --min_eta 0.4 --size_rel --lens_rim 5 --detect_sum 2"
sfx="c${corr}_pnr${pnr}_snr2_eta0.4_sz_rim5_sum3"
log_dir="$WORK_ROOT/logs_$sfx"
export FIG_DIR="$WORK_ROOT/cell_overlays_$sfx"
cd $work_dir
export PATH="$HOME/miniforge3/envs/parker/bin:$PATH"
export N_PROCESSES=8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for session in Clozapine/Vehicle/20191205_m85_clo01_amph Clozapine/LowDose/20191206_m85_clo05_amph; do
  for rec in 0 1; do
    name="$(basename $session)_rec$rec"
    ( python pipeline_raw_spikes.py "$session" $rec $corr $pnr $sel > "$log_dir/$name.log" 2>&1
      python compare_cell_extraction.py "$session" $rec "$WORK_ROOT" $corr $pnr $sel > "$log_dir/${name}_overlay.log" 2>&1
      echo "Finished $name" ) &
  done
done
wait
python collect_raw_pipeline_grid.py "$WORK_ROOT" 2_sum3 > "$WORK_ROOT/summary_$sfx.txt"
echo "All m85 jobs finished."
