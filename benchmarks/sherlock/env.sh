# Sourced by job.sh: the software environment every Sherlock benchmark job uses.
# $HOME is small, so conda envs live in $GROUP_HOME and package caches in
# $SCRATCH (regenerable; the JIT caches go node-local in job.sh).
export CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-$GROUP_HOME/$USER/envs}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$SCRATCH/cache/conda_pkgs}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/cache/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH/cache/pip}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH/cache/xdg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRATCH/cache/matplotlib}"
# Modules from the login profile (python, ...) would shadow the env on PATH.
# The purge leaves CentOS 7's gcc 4.8 (C89), which can't build _csmo.c
module purge 2>/dev/null || true
module load gcc/14.2.0
# shellcheck disable=SC1091
source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
LITE_ENV="${LITE_ENV:-gkmsvm-bench}"
LSGKM_ENV="${LSGKM_ENV:-lsgkm}"
# LS-GKM source builds: the kundajelab fork (gkmexplain) and mLS-GKM
LSGKM_BIN="${LSGKM_BIN:-$GROUP_HOME/$USER/src/lsgkm/bin}"
MLSGKM_BIN="${MLSGKM_BIN:-$GROUP_HOME/$USER/src/mLS-GKM/bin}"
