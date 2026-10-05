#!/usr/bin/env bash
# Self-contained timing benchmark for a remote GPU workstation.
#
# Bootstraps everything from scratch:
#   1. Creates conda envs (lsgkm + gkmsvm-lite)
#   2. Clones the repo
#   3. Downloads dsQTL training data
#   4. Runs the full timing comparison
#   5. Saves results + plot
#
# Usage:
#   scp benchmarks/run_timing_remote.sh user@server:~/
#   ssh user@server
#   bash run_timing_remote.sh              # full sweep with GPU
#   bash run_timing_remote.sh --quick      # small sweep for validation
#   bash run_timing_remote.sh --cpu-only   # no GPU column
#
# Prerequisites: conda or mamba, git, CUDA toolkit (for GPU)
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────

WORKDIR="${HOME}/gkmsvm_bench"
REPO_URL="https://github.com/adamyhe/gkmsvm-lite.git"
REPO_BRANCH="main"
LSGKM_ENV="lsgkm"
LITE_ENV="gkmsvm-bench"
QUICK=false
CPU_ONLY=false
SIZES="500,1000,2000,5000,10000,20000"
QUICK_SIZES="500,1000,2000"

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)     QUICK=true; shift ;;
        --cpu-only)  CPU_ONLY=true; shift ;;
        --workdir)   WORKDIR="$2"; shift 2 ;;
        --sizes)     SIZES="$2"; shift 2 ;;
        --branch)    REPO_BRANCH="$2"; shift 2 ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

if $QUICK; then
    SIZES="$QUICK_SIZES"
fi

DEVICE="cuda"
if $CPU_ONLY; then
    DEVICE="cpu"
fi

# Prefer mamba over conda for speed
CONDA=conda
if command -v mamba &>/dev/null; then
    CONDA=mamba
fi

echo "============================================================"
echo "gkmsvm-lite Timing Benchmark (remote)"
echo "============================================================"
echo "  Work dir:  $WORKDIR"
echo "  Device:    $DEVICE"
echo "  Sizes:     $SIZES"
echo "  Conda:     $CONDA"
echo "  Quick:     $QUICK"
echo ""

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# ── 1. Clone / update repo ────────────────────────────────────────

if [[ -d gkmsvm-lite/.git ]]; then
    echo ">>> Updating existing repo..."
    cd gkmsvm-lite
    git fetch origin
    git checkout "$REPO_BRANCH"
    git pull origin "$REPO_BRANCH"
    cd ..
else
    echo ">>> Cloning gkmsvm-lite..."
    git clone --branch "$REPO_BRANCH" "$REPO_URL"
fi

REPO="$WORKDIR/gkmsvm-lite"

# ── 2. Create LS-GKM conda env ────────────────────────────────────

if conda env list 2>/dev/null | grep -q "^${LSGKM_ENV} "; then
    echo ">>> LS-GKM env '$LSGKM_ENV' exists, skipping creation."
else
    echo ">>> Creating LS-GKM env '$LSGKM_ENV'..."
    $CONDA create -n "$LSGKM_ENV" -c bioconda -c conda-forge ls-gkm -y
fi

GKMTRAIN=$(conda run -n "$LSGKM_ENV" which gkmtrain 2>/dev/null || true)
if [[ -z "$GKMTRAIN" ]]; then
    echo "ERROR: gkmtrain not found in env '$LSGKM_ENV'"
    exit 1
fi
echo ">>> gkmtrain: $GKMTRAIN"

# ── 3. Create gkmsvm-lite conda env ───────────────────────────────

if conda env list 2>/dev/null | grep -q "^${LITE_ENV} "; then
    echo ">>> gkmsvm-lite env '$LITE_ENV' exists, reinstalling package..."
    conda run -n "$LITE_ENV" pip install -e "$REPO[dev,gpu]" --quiet 2>&1 | tail -3
else
    echo ">>> Creating gkmsvm-lite env '$LITE_ENV'..."
    $CONDA create -n "$LITE_ENV" python=3.11 -y
    conda run -n "$LITE_ENV" pip install -e "$REPO[dev,gpu]" 2>&1 | tail -5
fi

# Verify import
conda run -n "$LITE_ENV" python -c "import gkmsvm; print(f'gkmsvm-lite OK: {gkmsvm.__file__}')"

# Check GPU availability
if [[ "$DEVICE" == "cuda" ]]; then
    if conda run -n "$LITE_ENV" python -c "import cupy; cupy.cuda.Device(0).compute_capability" 2>/dev/null; then
        echo ">>> CuPy GPU OK"
    else
        echo "WARNING: CuPy not available, falling back to CPU"
        DEVICE="cpu"
    fi
fi

# ── 4. Download dsQTL data ─────────────────────────────────────────

DATA_DIR="$REPO/examples/data"
SEQ_DIR="$DATA_DIR/gm12878_sequence_sets"

if [[ -f "$SEQ_DIR/gm12878_shared.fa" ]]; then
    echo ">>> dsQTL data already present."
else
    echo ">>> Downloading dsQTL data..."
    mkdir -p "$DATA_DIR"
    TARBALL="$DATA_DIR/gm12878_sequence_sets.tar.gz"
    if [[ ! -f "$TARBALL" ]]; then
        curl -L -o "$TARBALL" \
            "https://beerlab.org/deltasvm/downloads/gm12878_sequence_sets.tar.gz"
    fi
    tar -xzf "$TARBALL" -C "$DATA_DIR"
    echo ">>> Extracted to $SEQ_DIR"
fi

N_POS=$(grep -c "^>" "$SEQ_DIR/gm12878_shared.fa")
echo ">>> $N_POS positive sequences available"

# ── 5. Run benchmark ──────────────────────────────────────────────

RESULTS_DIR="$WORKDIR/results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="$RESULTS_DIR/timing_vs_lsgkm_${TIMESTAMP}.tsv"
LOG="$RESULTS_DIR/timing_vs_lsgkm_${TIMESTAMP}.log"

echo ""
echo "============================================================"
echo "Running benchmark..."
echo "  Sizes:   $SIZES"
echo "  Device:  $DEVICE"
echo "  Output:  $OUTPUT"
echo "  Log:     $LOG"
echo "============================================================"
echo ""

conda run -n "$LITE_ENV" python "$REPO/benchmarks/timing_vs_lsgkm.py" \
    --conda-env "$LSGKM_ENV" \
    --device "$DEVICE" \
    --sizes "$SIZES" \
    --n-test 1000 \
    --repeats 3 \
    --output "$OUTPUT" \
    --plot \
    2>&1 | tee "$LOG"

echo ""
echo "============================================================"
echo "Benchmark complete!"
echo "============================================================"
echo "  Results: $OUTPUT"
echo "  Plot:    ${OUTPUT%.tsv}.png"
echo "  Log:     $LOG"
echo ""
echo "To copy results back:"
echo "  scp $(hostname):$OUTPUT ."
echo "  scp $(hostname):${OUTPUT%.tsv}.png ."
echo "  scp $(hostname):${OUTPUT%.tsv}.pdf ."
