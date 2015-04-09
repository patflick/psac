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

#PBS -lnodes=64:ppn=16:compute,walltime=1:00:00

source /home/pflick/gcc48env.sh

MPIRUN=/usr/mpi/gcc/mvapich2-1.7-qlc/bin/mpirun

# Change to directory from which qsub command was issued
cd $PBS_O_WORKDIR

# define executable to run
#EXE=./release/bin/test_sac
#EXE=./release/bin/benchmark_sac
EXE=./release/bin/benchmark_k

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
#for i in `seq 1 10`
#do
	#for p in # 64 128  #128 256 #512 1024
for i in `seq 1 10`
do
for k in 1 2 4 8 12 16 20
do
	for p in 512 1024
	do
		printf "### Running iteration $i with p = $p , k = $k processors ###" 1>&2
		#printf "$p;" 1>&2
		#/usr/bin/time -f "%E" $MPIRUN -np $p -errfile-pattern=$LOG_FOLDER/err-$p-%r.log -outfile-pattern=$LOG_FOLDER/out-$p-%r.log $EXE -f $INFILE -i 5
		$MPIRUN -np $p $EXE -f $INFILE -k $k
	done
done
done
#done
