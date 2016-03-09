#!/bin/sh

# NOTE: just change the processor number here to something power of 2 plus 4

#PBS -lnodes=72:ppn=16:compute,walltime=0:30:00
NUM_VOTE_OFF=8

#PBS -o BATCH_OUTPUT
#PBS -e BATCH_ERRORS

# set up compiler and MPI runtime environment
#source /home/pflick/gcc48env.sh
module load compilers/gcc-5.2
MPIRUN=/usr/mpi/gcc/mvapich2-1.7-qlc/bin/mpirun

BIN_FOLDER=./release/bin

# Change to directory from which qsub command was issued
cd $PBS_O_WORKDIR

# define executable to run
#EXE=./release/bin/benchmark_sac
EXE=$BIN_FOLDER/psac
BM=$BIN_FOLDER/mxx-bm-vote-off

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

NAME="BM-all2all-char-$PBS_NUM_NODES"

NUM_NODES=$(cat $PBS_NODEFILE | wc -l)
echo "[$NOW]: $NAME, exe=$EXE, file=$INFILE, nnodes=$NUM_NODES, ppn=$PBS_NUM_PPN, jobid=$PBS_JOBID, nodes:" >> $OUT_FOLDER/jobs.log
MY_NODES=$(cat $PBS_NODEFILE | sort -u | tr '\n' ', ')
echo "      $MY_NODES" >> $OUT_FOLDER/jobs.log
echo ""

# Old num nodes and PPN
PPN=$PBS_NUM_PPN
NUMNODES=$PBS_NUM_NODES
NEW_NUMNODES=`expr $NUMNODES - $NUM_VOTE_OFF`
NP=$(expr $NUMNODES \* $PPN)

#HOSTFILE=$PBS_NODEFILE
NEW_HOSTFILE=new_nodefile_$NEW_NUMNODES.txt
# run all-pairs bandwidth benchmark and exclude $NUM_VOTE_OFF worst nodes from the next job
$MPIRUN -np $NP -errfile-pattern=$LOG_FOLDER/bm-err-$NP-%r.log -outfile-pattern=$LOG_FOLDER/bm-out-$NP-%r.log $BM $NUM_VOTE_OFF $NEW_HOSTFILE


#HOSTFILE=$PBS_NODEFILE
HOSTFILE=$NEW_HOSTFILE

NUM_NODES=$(cat $HOSTFILE | wc -l)
NP=$(expr $NUM_NODES \* $PPN)
NP2=$(expr $NP / 2)

for p in $NP $NP2
#for p in 1024 512 256
#for p in 1600 1280 1024
#for p in 256 128
do
	for i in 1 2 3 4 5
	do
		printf "### Running psac iteration $i with p = $p processors ###\n" 1>&2
		/usr/bin/time $MPIRUN -np $p -hostfile $HOSTFILE -ppn $PPN -errfile-pattern=$LOG_FOLDER/err-$p-$i-%r.log -outfile-pattern=$LOG_FOLDER/out-$p-$i-%r.log $EXE -t -f $INFILE
		#printf "### Running psac iteration $i with p = $p processors and LCP ###\n" 1>&2
		#$MPIRUN -np $p -errfile-pattern=$LOG_FOLDER/err-$p-$i-%r.log -outfile-pattern=$LOG_FOLDER/out-$p-$i-%r.log $EXE -t -r 1000000
		#/usr/bin/time $MPIRUN -np $p -errfile-pattern=$LOG_FOLDER/err-$p-$i-lcp-%r.log -outfile-pattern=$LOG_FOLDER/out-$p-$i-lcp-%r.log $EXE --lcp -f $INFILE
	done
done
