#!/bin/bash
# run_fuzz_threads.sh PYTHON START END NTHREADS NOPS OUTDIR
PY=$1; START=$2; END=$3; NT=$4; NOPS=$5; OUT=$6
mkdir -p $OUT
for s in $(seq $START $END); do
  ulimit -v 8000000; timeout 900 $PY -W ignore fuzz_threads.py $s $NT $NOPS $OUT/log_$s > $OUT/stdout_$s.txt 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "seed=$s rc=$rc" >> $OUT/crashes.txt; fi
done
echo "finished $START-$END" >> $OUT/crashes.txt
