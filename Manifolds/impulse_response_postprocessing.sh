#!/bin/bash

# set condition
p1="Delta"
p2="alpha"
path="/home/rgf3807/data/ir_${p1}_${p2}"
n=77
batch_size=5
range_end=$(($n-1))

# execute python scripts in batches of batch_size
for IDX in `seq 0 $range_end`; do

  # python calls
  (
 	python function_generation_snn.py $IDX $p1 $p2 $path
 	python training_impulse_response.py $IDX $p1 $p2 $path
 	sleep $(( (RANDOM % 3) + 1))
 	) &

 	# batch control
 	if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
        wait -n
  fi

  # display progress
  echo "Finished ${IDX+1} of ${n} jobs."
done

wait
echo "All jobs finished."
