#!/usr/bin/env bash
# One Sherlock benchmark job, submitted by submit.sh (which sets the hardware):
#   job.sh setup [s3b]          envs, install, tests, data downloads
#   job.sh cpu|gpu <targets>    timing parts, on the fixed CPU or GPU hardware
#   job.sh merge <panels>       merge each panel's CPU and GPU parts
#   job.sh other <targets>      panels that time nothing (agreement, AUPRC, ...)
# Results go to $BENCH_REPO/benchmarks/results/, on the filesystem the checkout is on.
set -euo pipefail
: "${BENCH_REPO:?set by submit.sh}"
HW=$1
shift
# shellcheck source=env.sh
source "$BENCH_REPO/benchmarks/sherlock/env.sh"
export BENCH_HW=$HW
# conda's activation scripts read unset variables
activate() { set +u; conda activate "$1"; set -u; }

# Node-local temporary files and JIT caches: timed runs do no shared-filesystem
# I/O, and each job compiles its kernels in its own untimed warm-up rather than
# loading whatever another node or an older commit left in a shared cache
LOCAL="${L_SCRATCH_JOB:-${L_SCRATCH:-/tmp}/$SLURM_JOB_ID}"
export TMPDIR="$LOCAL/tmp" NUMBA_CACHE_DIR="$LOCAL/numba" CUPY_CACHE_DIR="$LOCAL/cupy"
mkdir -p "$TMPDIR" "$NUMBA_CACHE_DIR" "$CUPY_CACHE_DIR"

# Result headers record the hardware request (Slurm exports no variable for it)
if [[ -n ${SLURM_JOB_ID:-} ]]; then
    JOBINFO=$(scontrol show job "$SLURM_JOB_ID")
    BENCH_CONSTRAINT=$(grep -oP '\bFeatures=\K\S+' <<<"$JOBINFO" | grep -v '^(null)$' || true)
    BENCH_EXCLUSIVE=$(grep -q 'OverSubscribe=NO' <<<"$JOBINFO" && echo 1 || echo 0)
    export BENCH_CONSTRAINT BENCH_EXCLUSIVE
fi
echo ">>> $(hostname) job ${SLURM_JOB_ID:-?} ($HW): $*"
echo ">>> partition ${SLURM_JOB_PARTITION:-?}, constraint '${BENCH_CONSTRAINT:-}'," \
    "exclusive ${BENCH_EXCLUSIVE:-0}, $(nproc) CPUs"
lscpu | grep -E "^(Model name|Socket|Core|Thread|CPU MHz|CPU max MHz)" || true
if [[ $HW == gpu ]]; then
    nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm \
        --format=csv,noheader
fi
date

if [[ $HW == setup ]]; then
    # Envs (created in $CONDA_ENVS_PATH if missing), then this checkout into the
    # gkmsvm-lite env, editable, with uv
    if ! conda env list | grep -q "/$LSGKM_ENV\$"; then
        conda create -y -n "$LSGKM_ENV" -c bioconda -c conda-forge ls-gkm
    fi
    if ! conda env list | grep -q "/$LITE_ENV\$"; then
        conda create -y -n "$LITE_ENV" -c conda-forge python=3.11 uv
    fi
    activate "$LITE_ENV"
    command -v uv >/dev/null || conda install -y -n "$LITE_ENV" -c conda-forge uv
    uv pip install --python "$(command -v python)" -e "$BENCH_REPO[examples,gpu]" pytest
    # setup.py skips the SMO C extension when it fails to compile; fail here instead
    python -c "import gkmsvm._csmo"
    python -c "import gkmsvm, os; print('gkmsvm from', os.path.dirname(gkmsvm.__file__))"
    pytest -q -x -p no:cacheprovider "$BENCH_REPO/tests"
    cd "$BENCH_REPO/benchmarks"
    make data
    # S3B's data (downloads hg19) only when asked for
    if [[ ${1:-} == s3b ]]; then make data/ctcf/train_pos.fa; fi
    exit 0
fi

activate "$LITE_ENV"
# The env's editable install must be this checkout (setup makes it so)
python - "$BENCH_REPO" <<'EOF'
import os, sys
import gkmsvm, gkmsvm._csmo
here = os.path.realpath(os.path.dirname(gkmsvm.__file__))
want = os.path.realpath(os.path.join(sys.argv[1], "src", "gkmsvm"))
sys.exit(None if here == want else f"gkmsvm imports from {here}, not {want}: run setup")
EOF
cd "$BENCH_REPO/benchmarks"
DEVICE=cpu
[[ $HW == gpu || $HW == other ]] && DEVICE=cuda
MAKE_VARS=(SPLIT=1 DEVICE="$DEVICE" THREADS="${THREADS:-16}" LSGKM_ENV="$LSGKM_ENV")
[[ -d $LSGKM_BIN ]] && MAKE_VARS+=(LSGKM_BIN="$LSGKM_BIN")
[[ -d $MLSGKM_BIN ]] && MAKE_VARS+=(MLSGKM_BIN="$MLSGKM_BIN")
[[ -n ${QUICK:-} ]] && MAKE_VARS+=(QUICK=1)
# Panels that time nothing are independent: one failing doesn't stop the rest
[[ $HW == other ]] && MAKE_VARS+=(-k)
echo ">>> make $* ${MAKE_VARS[*]}"
make "$@" "${MAKE_VARS[@]}"
date
