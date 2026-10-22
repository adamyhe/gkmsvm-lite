#!/usr/bin/env bash
# Run paper benchmark targets from an existing gkmsvm-lite checkout.
#
# Creates the LS-GKM and gkmsvm-lite conda envs if they don't exist, installs
# this checkout into the gkmsvm-lite env, then runs `make -C benchmarks
# <targets>` inside it. Check out the commit you want first; the script doesn't
# touch git. Results land in benchmarks/results/, logs in benchmarks/results/logs/.
#
# Usage (from anywhere; the repo is located from this script's path):
#   bash benchmarks/run.sh                    # make paper (DEVICE=cuda)
#   bash benchmarks/run.sh fig1b s2           # selected targets
#   bash benchmarks/run.sh --quick            # small settings, results/quick/
#   bash benchmarks/run.sh --cpu-only fig1b
#   bash benchmarks/run.sh --lsgkm-bin <lsgkm>/bin fig1c   # source build with gkmexplain
#   bash benchmarks/run.sh --mlsgkm-bin <mLS-GKM>/bin       # add the mLS-GKM baseline
#
# Prerequisites: conda or mamba, make, CUDA driver (for GPU)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LSGKM_ENV="lsgkm"
LITE_ENV="gkmsvm-bench"
DEVICE="cuda"
MAKE_VARS=()
TARGETS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)     MAKE_VARS+=(QUICK=1); shift ;;
        --cpu-only)  DEVICE="cpu"; shift ;;
        --lsgkm-bin) MAKE_VARS+=("LSGKM_BIN=$2"); shift 2 ;;
        --mlsgkm-bin) MAKE_VARS+=("MLSGKM_BIN=$2"); shift 2 ;;
        --lsgkm-env) LSGKM_ENV="$2"; shift 2 ;;
        --env)       LITE_ENV="$2"; shift 2 ;;
        -*)          echo "Unknown option: $1"; exit 1 ;;
        *)           TARGETS+=("$1"); shift ;;
    esac
done
[[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=(paper)

CONDA=conda
command -v mamba &>/dev/null && CONDA=mamba

COMMIT="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [[ -n "$(git -C "$REPO" status --porcelain --untracked-files=no 2>/dev/null)" ]]; then
    echo "WARNING: $REPO has uncommitted changes; results will be tagged ${COMMIT}-dirty"
fi
echo ">>> Repo: $REPO @ $COMMIT"

# ── Conda envs ─────────────────────────────────────────────────────

if ! conda env list 2>/dev/null | grep -q "^${LSGKM_ENV} "; then
    echo ">>> Creating LS-GKM env '$LSGKM_ENV'"
    $CONDA create -n "$LSGKM_ENV" -c bioconda -c conda-forge ls-gkm -y
fi
conda run -n "$LSGKM_ENV" which gkmtrain >/dev/null \
    || { echo "ERROR: gkmtrain not found in env '$LSGKM_ENV'"; exit 1; }

if ! conda env list 2>/dev/null | grep -q "^${LITE_ENV} "; then
    echo ">>> Creating gkmsvm-lite env '$LITE_ENV'"
    $CONDA create -n "$LITE_ENV" -c conda-forge python=3.11 uv -y
fi
# uv, not pip (it also ignores pip.conf and PIP_* settings)
conda run -n "$LITE_ENV" which uv >/dev/null 2>&1 \
    || $CONDA install -n "$LITE_ENV" -c conda-forge uv -y
EXTRAS="examples"
[[ "$DEVICE" == "cuda" ]] && EXTRAS="examples,gpu"
conda run -n "$LITE_ENV" uv pip install --quiet -e "$REPO[$EXTRAS]"

if [[ "$DEVICE" == "cuda" ]] && \
   ! conda run -n "$LITE_ENV" python -c "import cupy; cupy.cuda.Device(0).compute_capability" 2>/dev/null; then
    echo "WARNING: CuPy cannot see a GPU; running on CPU"
    DEVICE="cpu"
fi

# ── Run ────────────────────────────────────────────────────────────

LOG_DIR="$REPO/benchmarks/results/logs"
mkdir -p "$LOG_DIR"
TAG="$(IFS=_; echo "${TARGETS[*]}")"
LOG="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_${COMMIT}_${TAG}.log"

echo ">>> make ${TARGETS[*]} DEVICE=$DEVICE ${MAKE_VARS[*]:-}  (log: $LOG)"
conda run --no-capture-output -n "$LITE_ENV" \
    make -C "$REPO/benchmarks" "${TARGETS[@]}" DEVICE="$DEVICE" LSGKM_ENV="$LSGKM_ENV" \
    ${MAKE_VARS[@]+"${MAKE_VARS[@]}"} 2>&1 | tee "$LOG"
