#!/bin/sh

#PBS -q swarm
#PBS -l nodes=1:ppn=16
#PBS -l walltime=0:30:00

# set up env
module purge
module load gcc/4.9.0
module load mvapich2/2.2
#module load openmpi/1.8
#module load impi/5.1.1.109
#module load openmpi


#BIN_FOLDER=./release-ompi/bin
BIN_FOLDER=./release3/bin
#BIN_FOLDER=./debug/bin

#GPFS=/gpfs/pace1/project/cse-aluru/pflick3

# Change to directory from which qsub command was issued
cd $PBS_O_WORKDIR

EXE=$BIN_FOLDER/desa-main

# prepare log folder
OUT_FOLDER=runlog
NOW=$(date +"%F_%H%M")
LOG_FOLDER=$OUT_FOLDER/$NOW
mkdir -p $LOG_FOLDER
rm $OUT_FOLDER/newest
ln -s $NOW $OUT_FOLDER/newest

# input args for executable
INFILE=$HOME/data/human_g1k_v37.actg
PATTERN_FILE=$HOME/data/patterns_human_32M_20.txt

#INFILE=$HOME/data/human_g1k_v37_chr22.actg
#PATTERN_FILE=$HOME/data/chr22_patterns.txt

NAME="desa-$PBS_NUM_NODES"

NUM_NODES=$(cat $PBS_NODEFILE | wc -l)
echo "[$NOW]: $NAME, exe=$EXE, file=$INFILE, nnodes=$PBS_NUM_NODES, ppn=$PBS_NUM_PPN, jobid=$PBS_JOBID, nodes:" >> $OUT_FOLDER/jobs.log
MY_NODES=$(cat $PBS_NODEFILE | sort -u | tr '\n' ', ')
cat $PBS_NODEFILE > $OUT_FOLDER/latest-nodefile.txt
echo "      $MY_NODES" >> $OUT_FOLDER/jobs.log
echo ""

# Old num nodes and PPN
PPN=$PBS_NUM_PPN
NUM_NODES=$PBS_NUM_NODES
NP=$(expr $NUM_NODES \\* $PPN)
NP2=$(expr $NP / 2)


echo "MPI is:"
echo `which mpirun`
echo `which mpiexec`
echo `which mpirun_rsh`

#-outfile-pattern=${LOG_FOLDER}/stdout-%r.log -errfile-pattern=${LOG_FOLDER}/stderr-%r.log 

#mpirun -np $p -ppn $PPN gdb -batch -ex run -ex bt $EXE $INFILE $PATTERN_FILE
#mpiexec -np $p -ppn $PPN $EXE $INFILE $PATTERN_FILE

TIMING_FILE="desa_timings_human32M_${NUM_NODES}_$(date +"%H%M").csv"

mpirun_rsh -rsh -np $NP -hostfile $PBS_NODEFILE $EXE $INFILE $PATTERN_FILE 2> $TIMING_FILE
#mpirun_rsh -rsh -np $NP -hostfile $PBS_NODEFILE gdb -batch -ex run -ex bt -ex "frame 5" -ex "print l" -ex "print r" --args $EXE $INFILE $PATTERN_FILE

#for p in $NP $NP2
#do
#	for i in 1 2 3
#	do
#		printf "### Running psac iteration $i with p = $p processors ###\n" 1>&2
#		mpirun -np $p -ppn $PPN $EXE $INFILE $PATTERN_FILE
#		#mpirun -np $p --map-by ppr:$PPN:node $EXE -f $INFILE
#	done
#done



