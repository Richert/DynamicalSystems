#!/bin/bash

# set condition
p1="Delta"
p2="alpha"
path="/home/rgf3807/data/ir_${p1}_${p2}"
n=77
batch_size=10
range_end=$(($n-1))

# limit amount of threads that each Python process can work with
n_threads=8
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads
export VECLIB_MAXIMUM_THREADS=$n_threads

# execute python scripts in batches of batch_size
for IDX in `seq 0 $range_end`; do

  # python calls
  (
  echo "Starting job #$(($IDX+1)) of ${n} jobs..."
  python function_generation_snn.py $IDX $p1 $p2 $path
  python training_impulse_response.py $IDX $p1 $p2 $path
  sleep 1
 	) &

 	# batch control
 	if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
        wait -n
  fi

done

wait
echo "All jobs finished."
