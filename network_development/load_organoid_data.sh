file_names=("LFP_Sp_160731")
file_ending = "mat"
url_address = "https://zenodo.org/records/4751759"
target_dir = "/media/richard/data/trujilo_2019"

for f in "${file_names}"; do
  curl --output "${target_dir}/${f}.${file_ending}" "${url_address}/${f}.${file_ending}?download=1"
done
