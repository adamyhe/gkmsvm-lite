#!/usr/bin/env bash
# Run all DART-Eval benchmarks for gkmsvm-lite.
#
# Prerequisites:
#   - DART-Eval data downloaded (set DART_WORK_DIR)
#   - Beer lab training data in examples/data/gm12878_sequence_sets/
#   - ENCODE models in benchmarks/data/models/
#   - pip install gkmsvm-lite[dev,gpu]
#
# Usage:
#   ./benchmarks/run_benchmarks.sh              # all benchmarks (GPU)
#   ./benchmarks/run_benchmarks.sh --device cpu  # CPU only
#   ./benchmarks/run_benchmarks.sh --quick       # quick validation run
set -euo pipefail

DEVICE="${DEVICE:-cuda}"
DART_WORK_DIR="${DART_WORK_DIR:-benchmarks/data/dart-eval}"
RESULTS_DIR="benchmarks/results"
MODELS_DIR="benchmarks/data/models"
QUICK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --device) DEVICE="$2"; shift 2 ;;
        --quick) QUICK=true; shift ;;
        --dart-dir) DART_WORK_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"
export DART_WORK_DIR

echo "============================================================"
echo "gkmsvm-lite DART-Eval Benchmark Suite"
echo "============================================================"
echo "  Device:    $DEVICE"
echo "  DART data: $DART_WORK_DIR"
echo "  Results:   $RESULTS_DIR"
echo "  Quick:     $QUICK"
echo ""

# ── 1. Lee 2015 dsQTL replication: train l=10k6 and l=11k7 models ──
echo "━━━ 1. dsQTL model training + VEP comparison ━━━"
TRAIN_ARGS=(
    --device "$DEVICE"
    --params both
    --output "$RESULTS_DIR/dsqtl_method_comparison.tsv"
    --save-models "$RESULTS_DIR/models"
)
if $QUICK; then
    TRAIN_ARGS+=(--n-negsets 1)
fi
python benchmarks/train_dsqtl_models.py "${TRAIN_ARGS[@]}"

# ── 2. Task 4: Chromatin activity with ENCODE models ──
echo ""
echo "━━━ 2. Task 4: Chromatin activity (ENCODE models) ━━━"

TASK4_MODELS=(
    # model_file                        cell_line  description
    "ENCFF953DDE.model.txt.gz           GM12878    ATAC_enhancer"
    "ENCFF701NCI.model.txt.gz           GM12878    DNase_enhancer"
)

# Add K562 if model exists
if [[ -f "$MODELS_DIR/ENCFF579AOX.model.txt.gz" ]]; then
    TASK4_MODELS+=("ENCFF579AOX.model.txt.gz K562 ATAC_enhancer")
fi

BATCH_SIZE=32
SV_CHUNK="8000"
if $QUICK; then
    BATCH_SIZE=16
fi

for entry in "${TASK4_MODELS[@]}"; do
    read -r model_file cell_line desc <<< "$entry"
    model_path="$MODELS_DIR/$model_file"
    if [[ ! -f "$model_path" ]]; then
        echo "  SKIP: $model_path not found"
        continue
    fi
    echo "  Running Task 4: $desc ($cell_line) with $model_file"
    python benchmarks/dart_eval.py \
        -m "$model_path" \
        --task activity \
        --cell-lines "$cell_line" \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE" \
        --sv-chunk-size "$SV_CHUNK" \
        --output "$RESULTS_DIR/task4_${cell_line}_${desc}.tsv" \
        -v
done

# ── 3. Task 5: VEP with ENCODE models ──
echo ""
echo "━━━ 3. Task 5: VEP with ENCODE models ━━━"

VEP_MODELS=(
    "ENCFF953DDE.model.txt.gz  GM12878_ATAC"
    "ENCFF701NCI.model.txt.gz  GM12878_DNase"
)

for entry in "${VEP_MODELS[@]}"; do
    read -r model_file desc <<< "$entry"
    model_path="$MODELS_DIR/$model_file"
    if [[ ! -f "$model_path" ]]; then
        echo "  SKIP: $model_path not found"
        continue
    fi
    # Run dsQTL and caQTL separately to avoid OOM on large variant sets
    for vep_task in dsqtl caqtl; do
        echo "  Running VEP ($vep_task): $desc with $model_file"
        python benchmarks/dart_eval.py \
            -m "$model_path" \
            --task vep \
            --vep-tasks "$vep_task" \
            --vep-method kernel \
            --device "$DEVICE" \
            --batch-size "$BATCH_SIZE" \
            --sv-chunk-size "$SV_CHUNK" \
            --output "$RESULTS_DIR/task5_${vep_task}_${desc}.tsv" \
            --save-scores "$RESULTS_DIR/scores_${desc}" \
            -v
    done
done

# ── 4. caQTL with ENCODE ATAC model ──
echo ""
echo "━━━ 4a. caQTL with ENCODE ATAC model ━━━"

if [[ -f "$MODELS_DIR/ENCFF953DDE.model.txt.gz" ]]; then
    python benchmarks/train_caqtl_model.py \
        --device "$DEVICE" \
        --model "$MODELS_DIR/ENCFF953DDE.model.txt.gz" \
        --batch-size "$BATCH_SIZE" \
        --output "$RESULTS_DIR/caqtl_encode_atac.tsv" \
        -v
else
    echo "  SKIP: ENCFF953DDE model not found"
fi

# ── 4b. caQTL with freshly trained GM12878 ATAC model ──
echo ""
echo "━━━ 4b. caQTL with trained GM12878 ATAC model ━━━"

CAQTL_TRAIN_ARGS=(
    --device "$DEVICE"
    --batch-size "$BATCH_SIZE"
    --output "$RESULTS_DIR/caqtl_trained.tsv"
    --save-model "$RESULTS_DIR/models/gm12878_atac_l11k7.npz"
    -v
)
if $QUICK; then
    CAQTL_TRAIN_ARGS+=(--max-train-seqs 5000)
fi

python benchmarks/train_caqtl_model.py "${CAQTL_TRAIN_ARGS[@]}"

echo ""
echo "============================================================"
echo "All benchmarks complete. Results in $RESULTS_DIR/"
echo "============================================================"
ls -la "$RESULTS_DIR/"
