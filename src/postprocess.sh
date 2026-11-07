#!/usr/bin/env bash
# Regenerate every paper artifact from results/ (analysis -> tables -> figures).
set -e
cd "$(dirname "$0")/.."
for exp in main scale momentum tau; do
  if [ -d "results/$exp" ] && [ "$(ls results/$exp | wc -l)" -gt 0 ]; then
    python src/analyze.py --exp "$exp" > "results/analysis_$exp.txt" 2>&1
  fi
done
python src/eda.py > /dev/null
python src/universal_lr.py > results/analysis_universal_lr.txt
python src/make_tables.py
python src/make_results_tables.py
python src/fig_onestep.py
python src/fig_sweep.py --exp main
python src/fig_extra.py
python src/fig_damping.py
echo "postprocess done"
