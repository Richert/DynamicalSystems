#!/bin/bash

# paths
work_dir="$HOME/PycharmProjects/DynamicalSystems/dimensionality"
save_dir="/gpfs/group/kennedy/rgast/results/dimensionality"

# modules and environments
cd $work_dir
source $HOME/.bashrc
module purge
mamba activate ds

# set condition
reversals=( -70.0 -65.0 -60.0 -55.0 -50.0 )
deltas=( 0.2 0.4 0.8 1.6 3.2 6.4 )
gs=( 0.0 4.0 8.0 12.0 16.0 20.0 )
n=20
batch_size=20
range_end=$((n-1))

# execute python scripts
counter=0
for r in "${reversals[@]}"; do
  for d in "${deltas[@]}"; do
    for g in "${gs[@]}"; do
      for IDX in $(seq 0 $range_end); do

        # process counter
        counter=$((counter+1))

        # python call
        (
        echo "Starting job #$((IDX+1)) of ${n} jobs for E_i = ${r}, g = ${g}, and delta = ${d}."
        srun --ntasks=1 --nodes=1 --mem=8G --time=00:30:00 --cpus-per-task=16 --job-name="dim_spn" \
        --output="out/dim_spn_$counter.out" --error="err/dim_spn_$counter.err" --partition="highcpu" --exclusive -c 1 \
        python simulation_dim_spn2.py "$save_dir" "$r" "$d" "$g" "$IDX"
        ) &

        # batch control
        if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
              wait
        fi

      done
    done
  done
done

wait

echo "All jobs finished."
cd $SLURM_SUBMIT_DIR
