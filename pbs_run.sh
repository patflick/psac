#!/bin/sh

#PBS -o BATCH_OUTPUT
#PBS -e BATCH_ERRORS
#PBS -lnodes=64:ppn=16:compute,walltime=0:30:00

# set up compiler and MPI runtime environment
source /home/pflick/gcc48env.sh
MPIRUN=/usr/mpi/gcc/mvapich2-1.7-qlc/bin/mpirun

# Change to directory from which qsub command was issued
cd $PBS_O_WORKDIR

# define executable to run
#EXE=./release/bin/benchmark_sac
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
#INFILE=$HOME/data/human_g1k_v37_chr22.actg
# srirams 5.7 GB file
# INFILE=/work/alurugroup/jnk/bhavani/data/SRR797058/half30m.fwd.txt

NAME="st-bulk-rma-human-big"

NUM_NODES=$(cat $PBS_NODEFILE | wc -l)
echo "[$NOW]: $NAME, exe=$EXE, file=$INFILE, nnodes=$NUM_NODES, ppn=$PBS_NUM_PPN, jobid=$PBS_JOBID, nodes:" >> $OUT_FOLDER/jobs.log
MY_NODES=$(cat $PBS_NODEFILE | sort -u | tr '\n' ', ')
echo "      $MY_NODES" >> $OUT_FOLDER/jobs.log
echo ""

	#for p in 8 16 32 64 128 256 
	#for p in 256 128 64
for i in 1 2 3 4 5
do
#for p in 64 128 256
#p=256

#for p in 512 256 128
for p in 1024 512 256
#for p in 1280 1600 1024
do
	printf "### Running psac iteration $i with p = $p processors ###\n" 1>&2
	/usr/bin/time $MPIRUN -np $p -errfile-pattern=$LOG_FOLDER/err-$p-$i-%r.log -outfile-pattern=$LOG_FOLDER/out-$p-$i-%r.log $EXE -t -f $INFILE
	#printf "### Running psac iteration $i with p = $p processors and LCP ###\n" 1>&2
	#$MPIRUN -np $p -errfile-pattern=$LOG_FOLDER/err-$p-$i-%r.log -outfile-pattern=$LOG_FOLDER/out-$p-$i-%r.log $EXE -t -r 1000000
	#/usr/bin/time $MPIRUN -np $p -errfile-pattern=$LOG_FOLDER/err-$p-$i-lcp-%r.log -outfile-pattern=$LOG_FOLDER/out-$p-$i-lcp-%r.log $EXE --lcp -f $INFILE
done
done
