#!/bin/bash

# set condition
neuron="fs"
path="/media/richard/data/snn_bifurcation"
n=20
batch_size=4
range_end=$(($n-1))

# limit amount of threads that each Python process can work with
n_threads=4
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
  python worker_snn_bifurcation.py $n $IDX $neuron $path
  sleep 1
 	) &

 	# batch control
 	if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
        wait -n
  fi

done

wait
echo "All jobs finished."
