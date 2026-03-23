#!/bin/bash
# Run all RL experiment variants, then generate visualization.
# Must be executed from the project root:
#   cd /path/to/AdaptiveBilevelOptimization
#   bash rl-experiment/run_rl.sh [N_STATES] [MAX_ITER_KL] [MAX_ITER_EUC]

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

N="${1:-10}"          # number of states
K_KL="${2:-200}"      # outer iterations for KL / mirror-descent
K_EUC="${3:-100}"     # outer iterations for Euclidean / projection

echo "============================================================"
echo " RL Experiment  |  n_states=$N  K_kl=$K_KL  K_euc=$K_EUC"
echo "============================================================"

echo ""
echo "--- KL / Mirror-descent : Adaptive BiOpt ---"
python rl-experiment/main.py \
    --n-states "$N" --adaptive --max-iter "$K_KL"

echo ""
echo "--- KL / Mirror-descent : Theory step-sizes ---"
python rl-experiment/main.py \
    --n-states "$N" --theory-steps --max-iter "$K_KL"

echo ""
echo "--- KL / Mirror-descent : Fixed step-sizes (gamma=0.5) ---"
python rl-experiment/main.py \
    --n-states "$N" --max-iter "$K_KL"

echo ""
echo "--- Euclidean / Projection : Adaptive BiOpt ---"
python rl-experiment/main_euclidean.py \
    --n-states "$N" --adaptive --max-iter "$K_EUC"

echo ""
echo "--- Euclidean / Projection : Fixed step-sizes ---"
python rl-experiment/main_euclidean.py \
    --n-states "$N" --max-iter "$K_EUC"

echo ""
echo "============================================================"
echo " All runs complete — generating visualization"
echo "============================================================"
python rl-experiment/visualize_rl.py
