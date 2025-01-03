#!/bin/bash

# set condition
deltas=( 0.0 1.0 2.0 4.0 8.0 )
gs=( 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 )
ps=( 0.05 0.1 0.2 )
n=10
batch_size=10
range_end=$((n-1))

# directories
save_dir="/media/richard/results/dimensionality"

# limit amount of threads that each Python process can work with
n_threads=9
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads
export VECLIB_MAXIMUM_THREADS=$n_threads

# execute python scripts in batches of batch_size
for p in "${ps[@]}"; do
  for d in "${deltas[@]}"; do
    for g in "${gs[@]}"; do
      for IDX in $(seq 0 $range_end); do

        # python calls
        (
        echo "Starting job #$((IDX+1)) of ${n} jobs for p = ${p}, g = ${g}, and delta = ${d}."
        python simulation_dim_exc.py $save_dir $p $d $g $IDX
        sleep 1
        ) &

        # batch control
        if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
              wait -n
        fi

      done
    done
  done
done

wait
echo "All jobs finished."
