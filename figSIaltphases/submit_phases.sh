#!/bin/bash 

# set nice descriptive name 
#$ -N figaltphases 
# use current working directory 
#$ -cwd 
# load current environment variables to context of the job
#$ -V 
# combine error and normal output into a single file 
#$ -j y 
# output in specified dir 
#$ -e logs 
#$ -o logs 
# declare the job to be not rerunable 
#$ -r n 
# run as an array job 
#$ -t 1-882
# specify queue to which to submit
#$ -q corei7b
# limit number of concurrent jobs
#$ -tc 50
# job priority in [-1023, 1024] (default = 0)
#$ -p -1

sleep $SGE_TASK_ID
echo $SGE_TASK_ID $HOSTNAME
python run_phases.py $SGE_TASK_ID
