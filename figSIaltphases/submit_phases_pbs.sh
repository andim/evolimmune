#!/bin/bash 

# set nice descriptive name 
#PBS -N SIalt 
# use current working directory 
#PBS -d .
# load current environment variables to context of the job
#PBS -V 
# combine error and normal output into a single file 
#PBS -j oe
# output in specified dir 
##PBS -e logs 
#PBS -o logs 
# declare the job to be not rerunable 
#PBS -r n 
# run as an array job (change number of tasks here)
#PBS -t 1-1008

echo $PBS_ARRAYID $HOSTNAME 
python run_phases.py $PBS_ARRAYID
