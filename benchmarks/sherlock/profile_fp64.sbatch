#!/usr/bin/env bash
# Profile the CUDA kernels for float64 use (dev/profile_fp64.py) on one GPU.
# Submit from the repo root, choosing the GPU:
#   sbatch -C GPU_SKU:L40S benchmarks/sherlock/profile_fp64.sbatch
# Output: benchmarks/results/logs/profile_fp64_<jobid>/ (git-ignored)
#SBATCH --job-name=gkm-prof-fp64
#SBATCH --partition=akundaje
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
set -uo pipefail
REPO="$SLURM_SUBMIT_DIR"
# shellcheck source=env.sh
source "$REPO/benchmarks/sherlock/env.sh"
module load cuda/12.6.1 # ncu, nsys
export TMPDIR="${L_SCRATCH_JOB:-/tmp}" CUPY_CACHE_DIR="${L_SCRATCH_JOB:-/tmp}/cupy"
OUT="$REPO/benchmarks/results/logs/profile_fp64_$SLURM_JOB_ID"
mkdir -p "$OUT"
cd "$REPO/benchmarks"
# nsys/ncu call the env's python directly: `conda run nsys ...` reads their
# options (--trace) as its own
PY="$(conda run -n "$LITE_ENV" which python)"
SCRIPT=dev/profile_fp64.py
OPS=$("$PY" -c "import sys; sys.path.insert(0, 'dev'); import profile_fp64; print(*profile_fp64.OPS)")
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo ">>> wall times"
"$PY" $SCRIPT --repeats 3 | tee "$OUT/times.txt"

echo ">>> nsys: kernel time per op"
nsys profile --trace=cuda,nvtx --force-overwrite=true -o "$OUT/nsys" "$PY" $SCRIPT
nsys stats --force-export=true --report cuda_gpu_kern_sum --report nvtx_kern_sum \
    --format csv --output "$OUT/nsys" "$OUT/nsys.nsys-rep"

# ncu: only the warm run (#1), one op per process: ISM and scoring launch
# hundreds of kernels, so a --launch-count shared by every op runs out early
NCU=(ncu --nvtx --nvtx-include "regex:#1/" --launch-count 300 --target-processes all)
METRICS=gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed
METRICS+=,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
METRICS+=,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active
METRICS+=,sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active
METRICS+=,sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active
METRICS+=,sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active
METRICS+=,sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active
METRICS+=,smsp__inst_executed.sum
METRICS+=,sm__sass_thread_inst_executed_op_dfma_pred_on.sum
METRICS+=,sm__sass_thread_inst_executed_op_dadd_pred_on.sum
METRICS+=,sm__sass_thread_inst_executed_op_dmul_pred_on.sum
METRICS+=,sm__sass_thread_inst_executed_op_integer_pred_on.sum
METRICS+=,lts__t_sectors_op_atom.sum,lts__t_sectors_op_red.sum
for op in $OPS; do
    echo ">>> ncu $op"
    "${NCU[@]}" --section SpeedOfLight --section ComputeWorkloadAnalysis \
        --section InstructionStats --csv --page details "$PY" $SCRIPT --ops "$op" \
        >"$OUT/ncu_sections_$op.csv"
    "${NCU[@]}" --metrics "$METRICS" --csv --page raw "$PY" $SCRIPT --ops "$op" \
        >"$OUT/ncu_metrics_$op.csv"
done
echo ">>> done: $OUT"
