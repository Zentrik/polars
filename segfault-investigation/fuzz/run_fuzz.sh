#!/bin/bash
# run_fuzz.sh PYTHON SCRIPT START END NOPS OUTDIR
PY=$1; SCRIPT=$2; START=$3; END=$4; NOPS=$5; OUT=$6
mkdir -p $OUT
for s in $(seq $START $END); do
  ulimit -v 6000000; timeout 600 $PY $SCRIPT $s $NOPS $OUT/log_$s.txt > $OUT/stdout_$s.txt 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "seed=$s rc=$rc" >> $OUT/crashes.txt; fi
done
echo "finished $START-$END" >> $OUT/crashes.txt
