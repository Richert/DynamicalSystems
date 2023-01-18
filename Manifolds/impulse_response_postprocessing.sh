#!/bin/sh

n_files = 77
for idx in seq 0 n_files-1
do
  python function_generation_snn.py idx
  python training_impulse_response.py idx
done
