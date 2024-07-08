#!/bin/bash

# set condition
deltas=( 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 )
gs=( 0.0 4.0 8.0 12.0 16.0 20.0 24.0 28.0 32.0 )
n=20
batch_size=10
range_end=$(($n-1))

# limit amount of threads that each Python process can work with
n_threads=9
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads
export VECLIB_MAXIMUM_THREADS=$n_threads

# execute python scripts in batches of batch_size
for d in "${deltas[@]}"; do
  for g in "${gs[@]}"; do
    for IDX in `seq 0 $range_end`; do

      # python calls
      (
      echo "Starting job #$(($IDX+1)) of ${n} jobs for g = ${g} and delta = ${d}."
      python simulation_inh_dim_ss.py $d $g $IDX
      sleep 1
      ) &

      # batch control
      if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
            wait -n
      fi

    done
  done
done

wait
echo "All jobs finished."
