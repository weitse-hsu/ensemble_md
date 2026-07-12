#!/bin/bash

#SBATCH --nodes 1
#SBATCH --partition=amilan
#SBATCH --ntasks=20
#SBATCH --job-name=tutorial
#SBATCH --qos=normal
#SBATCH --time=03:00:00
#SBATCH --output=tutorial.out

source /projects/anfr8476/pkgs/gromacs-2022.5-wl/bin/GMXRC
ml gcc
ml openmpi/5.0.6

conda activate EEXE

mpirun -np 5 run_REXEE -y params.yaml

mpirun -np 5 analyze_REXEE -y params.yaml