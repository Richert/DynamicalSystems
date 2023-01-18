#!/bin/bash

for IDX in $(seq 0 76)
do
 	python function_generation_snn.py $IDX
 	python training_impulse_response.py $IDX
done
