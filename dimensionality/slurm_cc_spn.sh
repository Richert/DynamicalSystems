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
deltas=( 0.5 1.0 1.5 2.0 2.5 3.0 )
gs=( 0.0 0.4 0.8 1.2 1.6 2.0 )
es=( -65.0 -60.0 -55.0 -50.0 )
n=20
batch_size=20
range_end=$((n-1))

# execute python scripts
counter=0
for e_r in "${es[@]}"; do
  for d in "${deltas[@]}"; do
    for g in "${gs[@]}"; do
      for IDX in $(seq 0 $range_end); do

        # process counter
        counter=$((counter+1))

        # python call
        (
        echo "Starting job #$((IDX+1)) of ${n} jobs for E_r = ${e_r}, g = ${g} and Delta = ${d}."
        srun --ntasks=1 --nodes=1 --mem=8G --time=00:45:00 --cpus-per-task=12 --job-name="cc_spn" \
        --output="out/cc_spn_$counter.out" --error="err/cc_spn_$counter.err" --partition="shared" --exclusive -c 1 \
        python simulation_cc_spn.py "$save_dir" "$e_r" "$d" "$g" "$IDX"
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
