#!/bin/bash -l
#$ -P my_project
#$ -l h_rt=02:00:00
#$ -pe omp 4
export OMP_NUM_THREADS=$NSLOTS
echo "ok"
