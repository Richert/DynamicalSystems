#!/bin/bash

# files, directories and names
file_names=( "LFP_Sp_160731" "LFP_Sp_160803" "LFP_Sp_160810" "LFP_Sp_160816" "LFP_Sp_160824" "LFP_Sp_160902" \
  "LFP_Sp_160902" "LFP_Sp_160913" "LFP_Sp_160916" "LFP_Sp_160920" "LFP_Sp_160923" "LFP_Sp_161001" "LFP_Sp_1610107" \
  "LFP_Sp_161008" "LFP_Sp_161014" "LFP_Sp_161018" "LFP_Sp_161021" "LFP_Sp_161028" "LFP_Sp_161104" "LFP_Sp_161110" \
  "LFP_Sp_161118" "LFP_Sp_161124" "LFP_Sp_161206" "LFP_Sp_161209" "LFP_Sp_161216" "LFP_Sp_161217" "LFP_Sp_161223" \
  "LFP_Sp_161230" "LFP_Sp_170106" "LFP_Sp_170113" "LFP_Sp_170120" "LFP_Sp_170203" "LFP_Sp_170207" "LFP_Sp_170210" \
  "LFP_Sp_170217" "LFP_Sp_170224" "LFP_Sp_170303" "LFP_Sp_170310" "LFP_Sp_170316" )
file_ending="mat"
url_address="https://zenodo.org/records/4751759/files"
target_dir="/home/richard/data/trujilo_2019"

# load data
for f in "${file_names[@]}"; do
  curl --output "${target_dir}/${f}.${file_ending}" "${url_address}/${f}.${file_ending}"
done
