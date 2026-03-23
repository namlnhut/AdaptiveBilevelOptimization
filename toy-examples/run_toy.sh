#!/bin/bash
# Run all toy.py configurations for Figure 1
# eta_correct = 2/20002 ~ 1e-4 (default)
# gamma_correct = 2/(L_f + H_f) = 2/2.1 ~ 0.95

set -e
cd "$(dirname "$0")"

echo "=== Adaptive BiOpt ==="
python toy.py --adaptive

echo "=== eta=1.1e-4 (misspecified), gamma=0.95 (correct) ==="
python toy.py --eta 1.1e-4 --gamma 0.95

echo "=== eta=2e-4 (misspecified), gamma=0.95 (correct) ==="
python toy.py --eta 2e-4 --gamma 0.95

echo "=== eta=correct (2/20002), gamma=1 (misspecified) ==="
python toy.py --gamma 1.0

echo "=== eta=correct (2/20002), gamma=0.96 (misspecified) ==="
python toy.py --gamma 0.96

echo ""
echo "=== All runs complete. Running visualization... ==="
python visualize.py
