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
deltas=( 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 )
gs=( 0.0 4.0 8.0 12.0 16.0 20.0 24.0 28.0 )
n=20
batch_size=20
range_end=$((n-1))

# execute python scripts
counter=0
for d in "${deltas[@]}"; do
  for g in "${gs[@]}"; do
    for IDX in $(seq 0 $range_end); do

      # python call
      (
      counter=$((counter+1))
      echo "Starting job #$((IDX+1)) of ${n} jobs for g = ${g} and delta = ${d}."
      srun --ntasks=1 --nodes=1 --mem=8G --time=00:30:00 --cpus-per-task=12 --job-name="dim_spn_$counter" \
      --output="out/dim_spn_$counter.out" --error="err/dim_spn_$counter.err" --partition="highcpu" --exclusive -c 1 \
      python simulation_dim_spn.py "$save_dir" "$d" "$g" "$IDX"
      ) &

      # batch control
      if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
            wait
      fi

    done
  done
done

wait

echo "All jobs finished."
cd $SLURM_SUBMIT_DIR
