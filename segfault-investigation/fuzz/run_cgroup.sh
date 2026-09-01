#!/bin/bash
# run_cgroup.sh PY LIMIT_BYTES NWORKERS ITERS OUTFILE [extra env]
PY=$1; LIMIT=$2; NW=$3; ITERS=$4; OUT=$5
CG=/sys/fs/cgroup/memory/plstress_$$
mkdir -p $CG && echo $LIMIT > $CG/memory.limit_in_bytes
echo $$ > $CG/cgroup.procs
export POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-16}
cd "$(dirname "$0")"
for w in $(seq 1 $NW); do
  ( $PY -W ignore over_stress.py run data $ITERS $OUT streaming pyarrow; echo "worker $w exit=$?" >> $OUT ) &
done
wait
echo "peak=$(cat $CG/memory.max_usage_in_bytes) failcnt=$(cat $CG/memory.failcnt)" >> $OUT
echo 0 > /sys/fs/cgroup/memory/cgroup.procs 2>/dev/null
