#!/bin/bash

# set condition
p1="Delta"
p2="alpha"
path="/home/rgf3807/data/ir_${p1}_${p2}"
n=77
range_end=$(($n-1))

# execute python scripts
for IDX in `seq 0 $range_end`;
do
 	python function_generation_snn.py $IDX $p1 $p2 $path
 	python training_impulse_response.py $IDX $p1 $p2 $path
done
