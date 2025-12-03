#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=48
#SBATCH --partition=genoa
#SBATCH --time=40:00:00
#SBATCH --job-name="qudit ruskai codes"
#SBATCH --output=data/outs/qudit_ruskai_%j.out
#SBATCH --error=data/outs/qudit_ruskai_%j.err.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=l.j.bond@uva.nl

# Load modules
module purge
module load 2024
module load Julia/1.10.4-linux-x86_64

for i in {1..48} 
do
	julia --project=. --threads=1 qudit_ruskai_cluster_single_core.jl $1 $SLURM_ARRAY_TASK_ID $i &
done
wait
