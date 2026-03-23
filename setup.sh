# 1. Create environment with a specific Python version (3.11 or 3.12 recommended for stability in 2026)
conda create -n benchopt python=3.11 -y
conda activate benchopt

# 2. Upgrade pip immediately (helps avoid many wheel/resolution issues)
pip install --upgrade pip setuptools wheel

# 3. Install benchopt first (it may pull jax as a dependency, but usually CPU-only)
pip install -U benchopt

# 4. Install JAX with CUDA 12 support (official easiest method – pulls nvidia-cu* packages automatically)
#    This is the currently recommended command per JAX docs
pip install --upgrade "jax[cuda12]"

# 5. Make sure LD_LIBRARY_PATH is NOT set (prevents system/HPC CUDA libs from overriding JAX's bundled ones)
#    → Do this AFTER jax install so it takes effect for future runs
unset LD_LIBRARY_PATH
# Optional: make it permanent in this env (reloads on every activation)
conda env config vars set LD_LIBRARY_PATH=""

# 6. Install scikit-learn (and upgrade if already present)
pip install -U scikit-learn

# 7. Optional but very useful: install a few common extras for benchopt + JAX workflows
pip install -U numpy scipy pandas matplotlib numba  # often useful for benchmarks

# 8. Quick verification
echo "JAX devices:"
python -c "import jax; print(jax.devices()); print(jax.devices('gpu')) if 'gpu' in jax.lib.xla_bridge.get_backend().platform else print('No GPU backend')"
echo -e "\nnvidia-smi output:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo -e "\ncuSPARSE check (should find something):"
ldconfig -p | grep -i cusparse || echo "No cuSPARSE → something is still wrong"