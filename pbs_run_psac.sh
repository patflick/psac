#!/bin/bash

# Instructions for new CyEnce Cluster users:
#  To use this script:
#   1) Save this script as a file named myscript on share
#   2) On share, Issue                   
#       qsub myscript    to submit the job 
#        Use qstat -a to see job status, 
#         Use qdel jobname to delete one of your jobs
#         jobnames are of the form 1234.share

###########################################
# Output goes to file BATCH_OUTPUT.
# Error output goes to file BATCH_ERRORS.
# If you want the output to go to another file, change BATCH_OUTPUT 
# or BATCH_ERRORS in the following lines to the full path of that file. 

#PBS  -o BATCH_OUTPUT 
#PBS  -e BATCH_ERRORS 

#PBS -lnodes=16:ppn=16:compute,walltime=1:00:00

MPIRUN=/usr/mpi/gcc/mvapich2-1.7-qlc/bin/mpirun

# Change to directory from which qsub command was issued 
cd $PBS_O_WORKDIR

# define executable to run
EXE=./build/bin/test_sac

# prepare log folder
OUT_FOLDER=runlog
NOW=$(date +"%F_%H%M")
LOG_FOLDER=$OUT_FOLDER/$NOW
mkdir -p $LOG_FOLDER
rm $OUT_FOLDER/newest
ln -s $NOW $OUT_FOLDER/newest

# input args for executable
INFILE=/lustre/alurugroup/pflick/human_g1k_v37.actg

#for p in 1 2 4 8 16 32 64 128 256 512 1024
for p in 64 128 256 #512 1024
do
	echo "### Running with p = $p processors ###"
	time $MPIRUN -np $p -errfile-pattern=$LOG_FOLDER/err-$p-%r.log -outfile-pattern=$LOG_FOLDER/out-$p-%r.log $EXE $INFILE
done
