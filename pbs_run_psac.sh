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

#PBS -lnodes=2:ppn=16:compute,walltime=4:00:00

source /home/pflick/gcc48env.sh

MPIRUN=/usr/mpi/gcc/mvapich2-1.7-qlc/bin/mpirun

# Change to directory from which qsub command was issued
cd $PBS_O_WORKDIR

# define executable to run
#EXE=./release/bin/test_sac
#EXE=./release/bin/benchmark_sac
#EXE=./release/bin/benchmark_k
EXE=./release/bin/psac

# prepare log folder
OUT_FOLDER=runlog
NOW=$(date +"%F_%H%M")
LOG_FOLDER=$OUT_FOLDER/$NOW
mkdir -p $LOG_FOLDER
rm $OUT_FOLDER/newest
ln -s $NOW $OUT_FOLDER/newest

# input args for executable
INFILE=/lustre/alurugroup/pflick/human_g1k_v37.actg
#INFILE=/lustre/alurugroup/pflick/Pabies10.actg
#INFILE=/lustre/alurugroup/pflick/human2g.actg

NAME="psac-human-compare-lcp-small"

echo "[$NOW]: $NAME, exe=$EXE, file=$INFILE, ppn=$PBS_NUM_PPN, jobid=$PBS_JOBID, nodes:" >> $OUT_FOLDER/jobs.log
MY_NODES=$(cat $PBS_NODEFILE | sort -u | tr '\n' ', ')
echo "      $MY_NODES" >> $OUT_FOLDER/jobs.log
echo ""

for i in `seq 1 10`
do
	#for p in 8 16 32 64 128 256 
	#for p in 256 128 64
	for p in 32 16 1
	#for p in 512 1024
	#for p in 1024 512 256 128 64
	#for p in 1280 1600 1024
	do
		printf "### Running psac iteration $i with p = $p processors ###\n" 1>&2
		/usr/bin/time $MPIRUN -np $p -errfile-pattern=$LOG_FOLDER/err-$p-$i-%r.log -outfile-pattern=$LOG_FOLDER/out-$p-$i-%r.log $EXE -f $INFILE
		printf "### Running psac iteration $i with p = $p processors and LCP ###\n" 1>&2
		/usr/bin/time $MPIRUN -np $p -errfile-pattern=$LOG_FOLDER/err-$p-$i-lcp-%r.log -outfile-pattern=$LOG_FOLDER/out-$p-$i-lcp-%r.log $EXE --lcp -f $INFILE
		#$MPIRUN -np $p $EXE -f $INFILE
	done
done
#done
