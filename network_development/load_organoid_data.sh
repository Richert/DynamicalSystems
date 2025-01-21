#!/bin/bash

# files, directories and names
file_names=( "LFP_Sp_160731" "LFP_Sp_160803" )
file_ending="mat"
url_address="https://zenodo.org/records/4751759"
target_dir="/media/richard/data/trujilo_2019"

# load data
for f in "${file_names[@]}"; do
  curl --output "${target_dir}/${f}.${file_ending}" "${url_address}/${f}.${file_ending}"
done
