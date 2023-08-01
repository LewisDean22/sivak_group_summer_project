#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --account=def-dsivak
#SBATCH --output=job_output.out
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2G
module purge && module load gcc/9.3.0 && module save
module load python/3.11.2
source virtual_environment_test/bin/activate
module load scipy-stack/2023a
module load StdEnv/2020
module load quast/5.0.2
python setup.py build_ext --inplace
python cutoff_frequency_analysis.py
