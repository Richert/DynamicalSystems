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
deltas=( 0.5 1.0 2.0 4.0 8.0 )
gs=( 0.0 0.3 0.6 0.9 1.2 1.5 1.8 2.1 )
ps=( 0.05 0.2 )
n=20
batch_size=10
range_end=$((n-1))

# execute python scripts
counter=0
for p in "${ps[@]}"; do
  for d in "${deltas[@]}"; do
    for g in "${gs[@]}"; do
      for IDX in $(seq 0 $range_end); do

        # process counter
        counter=$((counter+1))

        # python call
        (
        echo "Starting job #$((IDX+1)) of ${n} jobs for p = ${p}, g = ${g} and delta = ${d}."
        srun --ntasks=1 --nodes=1 --mem=8G --time=01:00:00 --cpus-per-task=1 --job-name="cc_exc" \
        --output="out/cc_exc_$counter.out" --error="err/cc_exc_$counter.err" --partition="gpu" --exclusive -c 1 \
        python simulation_cc_exc.py "$save_dir" "$p" "$d" "$g" "$IDX"
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
