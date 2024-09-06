#!/bin/bash

#SBATCH --job-name=dim_spn
#SBATCH --output=out/dim_spn.out
#SBATCH --error=err/dim_spn.err
#SBATCH --time=24:00:00
#SBATCH --partition=highcpu
#SBATCH --cpus-per-task=16
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=10G

# paths
work_dir="$HOME/PycharmProjects/DynamicalSystems/dimensionality"
save_dir="/gpfs/group/kennedy/rgast/results/dimensionality"

# modules and environments
cd $work_dir
source $HOME/.bashrc
module purge
mamba activate ds

# limit amount of threads that each Python process can work with
#n_threads=16
#export OMP_NUM_THREADS=$n_threads
#export OPENBLAS_NUM_THREADS=$n_threads
#export MKL_NUM_THREADS=$n_threads
#export NUMEXPR_NUM_THREADS=$n_threads
#export VECLIB_MAXIMUM_THREADS=$n_threads

# set condition
deltas=( 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 )
gs=( 0.0 4.0 8.0 12.0 16.0 20.0 24.0 28.0 )
n=20
batch_size=64
range_end=$((n-1))

# execute python scripts
for d in "${deltas[@]}"; do
  for g in "${gs[@]}"; do
    for IDX in $(seq 0 $range_end); do

      # python call
      (
      echo "Starting job #$((IDX+1)) of ${n} jobs for g = ${g} and delta = ${d}."
      srun python simulation_dim_spn.py "$save_dir" "$d" "$g" "$IDX"
      ) &

      # batch control
      if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
            wait
      fi

    done
  done
done

wait

echo "All jobs finished."
cd $SLURM_SUBMIT_DIR
