#!/bin/bash
# Run the experiment pipeline. Finished runs and steps are skipped, so it can be restarted.
#   bash run.sh            # full pipeline
#   SMOKE=1 bash run.sh    # unit tests plus 2-epoch runs of every method
# Settings: PYTHON (default python3), WORKERS (default 30, each a single-threaded run).
# All output is appended to run.out (smoke: smoke/run.out). CPU only by design (float64 exact solves).
set -u
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python  # avoids a protobuf/onnx mismatch in some images (torch._dynamo via Adam)
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3}
W=${WORKERS:-30}
[ -w "$HERE" ] || { echo "ERROR: $HERE is not writable for user $(id -un)"; exit 1; }

if [ "${SMOKE:-0}" = 1 ]; then
  OUT=$HERE/smoke/run.out; mkdir -p "$HERE/smoke"
else
  OUT=$HERE/run.out
fi
exec > >(tee -a "$OUT") 2>&1
TEE=$!
flushlog() { exec >&- 2>&-; for _ in $(seq 50); do kill -0 $TEE 2>/dev/null || break; sleep 0.1; done; }
trap flushlog EXIT

echo "===== $(date '+%F %T') launch on $(hostname) as $(id -un), workers=$W"
[ -e "$HERE/data/processed/har.npz" ] || { echo "ERROR: data/processed not found, run src/data_prep.py first (see README)"; exit 1; }
"$PY" -c "import torch, numpy; print('torch', torch.__version__, 'numpy', numpy.__version__)"

if [ "${SMOKE:-0}" = 1 ]; then
  RUNS=("test_core" "smoke")
else
  RUNS=("test_core" "onestep" "main" "scale" "momentum" "tau" "final_test" "logreg" "bench")
fi

run_one() {  # each step is idempotent: sweeps skip finished runs, the others check their output file
  local name=$1
  local RES=$HERE/results
  echo "----- $(date '+%F %T') $name: start"
  case $name in
    test_core)  "$PY" -u "$HERE/src/test_core.py" ;;
    onestep)    [ -e "$RES/onestep/records.json" ] && { echo "done before, skipped"; return 0; }
                OMP_NUM_THREADS=4 "$PY" -u "$HERE/src/onestep.py" && "$PY" -u "$HERE/src/dump_modes.py" ;;
    smoke)      "$PY" -u "$HERE/src/sweep.py" --exp smoke --workers "$W" ;;
    main)       "$PY" -u "$HERE/src/sweep.py" --exp main --workers "$W" ;;
    scale)      "$PY" -u "$HERE/src/sweep.py" --exp scale --workers "$W" --seeds 0 1 2 ;;
    momentum)   "$PY" -u "$HERE/src/sweep.py" --exp momentum --workers "$W" --seeds 0 1 2 ;;
    tau)        "$PY" -u "$HERE/src/sweep.py" --exp tau --workers "$W" --seeds 0 1 2 ;;
    final_test) [ -e "$RES/test_main_selection.json" ] && [ "$(ls "$RES/test_main" | wc -l)" -ge 270 ] \
                  && { echo "done before, skipped"; return 0; }
                "$PY" -u "$HERE/src/final_test.py" --exp main --workers "$W" ;;
    logreg)     [ -e "$RES/logreg.json" ] && { echo "done before, skipped"; return 0; }
                "$PY" -u "$HERE/src/logreg.py" ;;
    bench)      [ -e "$RES/bench_step.json" ] && { echo "done before, skipped"; return 0; }
                "$PY" -u "$HERE/src/bench_step.py" ;;
  esac
}

status=0
for r in "${RUNS[@]}"; do
  run_one "$r" || { status=$?; echo "----- $r: FAILED with exit $status"; break; }
  echo "----- $(date '+%F %T') $r: finished"
done
echo "===== $(date '+%F %T') exit $status"
exit $status
