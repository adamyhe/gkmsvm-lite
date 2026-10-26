#!/usr/bin/env bash
# Submit the application-note benchmarks to Sherlock's scheduler, then exit.
#
# Timing panels are split into a CPU part (LS-GKM, mLS-GKM, gkmsvm-lite CPU)
# and a GPU part (gkmsvm-lite CUDA), each on one fixed hardware type, so times
# compare across panels and reruns:
#   CPU parts  whole node (--exclusive) of CPU_CONSTRAINT in CPU_PARTITION, one
#              job each (no two share a node, so they may run at once)
#   GPU parts  one GPU of the --gpu type; chained, so two GPU parts never run
#              at once on one node
#   merge      per panel, once its parts succeed: results/<file>.tsv
# Panels that time nothing (fig1a fig1d s1 s4 s5) run as one GPU job, "other".
# A setup job (envs, install, tests, downloads) runs first.
#
#   bash benchmarks/sherlock/submit.sh                    # fig1b fig1c s2 s3
#   bash benchmarks/sherlock/submit.sh --quick            # small settings, results/quick/
#   bash benchmarks/sherlock/submit.sh --gpu a100 s2 other
#   bash benchmarks/sherlock/submit.sh --no-setup --dry-run fig1c
#
# Panels: fig1b fig1c s2 s3 s3b (LS-GKM up to a day per fit; not by default),
# other. Overrides (environment): PARTITION, CPU_PARTITION, CPU_CONSTRAINT,
# GPU_PARTITION, GPU_CONSTRAINT, GPU_CPUS, THREADS, OTHER_TARGETS.
set -euo pipefail
BENCH_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export BENCH_REPO

GPU=l40s SETUP=1 DRY=0 QUICK=
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU=$2; shift 2 ;;
        --no-setup) SETUP=0; shift ;;
        --quick) QUICK=1; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *) break ;;
    esac
done
PANELS=("$@")
[[ ${#PANELS[@]} -eq 0 ]] && PANELS=(fig1b fig1c s2 s3)
export QUICK

# ── Hardware ───────────────────────────────────────────────────────
# Everything runs in the group's partition, akundaje: jobs up to 7 days (normal
# allows 48 h) and short queues. CPU: AMD EPYC 9384X (64 cores, 384 GB; two
# such nodes), the whole node, so no other job shares its cores, caches or
# memory bandwidth; with two nodes, at most two CPU parts run at once.
PARTITION=${PARTITION:-akundaje}
CPU_PARTITION=${CPU_PARTITION:-$PARTITION}
CPU_CONSTRAINT=${CPU_CONSTRAINT:-CPU_SKU:9384X}
# GPU: the GPU and the host CPU, which runs the GPU part's host-side work
case $GPU in
    l40s) GP=$PARTITION GC="GPU_SKU:L40S&CPU_SKU:6426Y" ;;
    a100) GP=$PARTITION GC="GPU_SKU:A100_SXM4&GPU_MEM:80GB" ;;
    h100) GP=gpu GC="GPU_SKU:H100_SXM5&CPU_SKU:8462Y+" ;; # public gpu: 48 h at most
    *) echo "Unknown --gpu $GPU (l40s, a100, h100)"; exit 1 ;;
esac
GPU_PARTITION=${GPU_PARTITION:-$GP}
GPU_CONSTRAINT=${GPU_CONSTRAINT:-$GC}
# THREADS: LS-GKM -T (1, 4 or 16) and gkmsvm-lite's CPU threads, everywhere
export THREADS=${THREADS:-16}
GPU_CPUS=${GPU_CPUS:-$THREADS}

# Time limits, about 2-3x estimates, and GPU job host memory (S3/S3B hold the
# N x N Gram in host RAM). LS-GKM dominates the CPU parts. Lee 2016 (Fig. 1B,
# H1hescCtcf, 4-core 2016 i7) trained 90k in ~100 min (4 threads, 8 GB cache)
# to ~360 min (4 threads, 100 MB), and 20k in ~120 min on 1 thread: S3B's 16
# fits come to ~22 h at those speeds, 1B's 1-thread LS-GKM to ~2 h at N=20k.
# 1C keeps more room: no published time for single-threaded gkmexplain.
declare -A TIME=(
    [fig1b-cpu]=1-00:00:00 [fig1b-cpu-t1]=2-00:00:00 [fig1b-gpu]=12:00:00
    [fig1c-cpu]=3-00:00:00 [fig1c-gpu]=12:00:00
    [s2-cpu]=1-00:00:00 [s2-gpu]=12:00:00
    [s3-cpu]=1-00:00:00 [s3-cpu-lite]=1-00:00:00 [s3-gpu]=1-00:00:00
    [s3b-cpu]=3-00:00:00 [s3b-gpu]=2-00:00:00
    [other]=2-00:00:00
)
declare -A MEM=([s3-gpu]=128G [s3b-gpu]=200G [other]=96G)
# Each panel's jobs: CPU parts, GPU parts, and the targets the merge makes
declare -A CPU_UNITS=(
    [fig1b]="fig1b-cpu fig1b-cpu-t1" [fig1c]=fig1c-cpu [s2]=s2-cpu
    [s3]="s3-cpu s3-cpu-lite" [s3b]=s3b-cpu
)
declare -A GPU_UNITS=(
    [fig1b]=fig1b-gpu [fig1c]=fig1c-gpu [s2]=s2-gpu [s3]=s3-gpu [s3b]=s3b-gpu
)
OTHER_TARGETS=${OTHER_TARGETS:-fig1a fig1d s1 s4 s5}
# S5 needs DART-Eval's data, downloaded from Synapse with your account
# (dart_download.py); without it, leave S5 out rather than fail in the job
DART_DIR=${DART_WORK_DIR:-$BENCH_REPO/benchmarks/data/dart-eval}
if [[ " $OTHER_TARGETS " == *" s5 "* && ! -d $DART_DIR ]]; then
    OTHER_TARGETS=${OTHER_TARGETS/s5/}
    echo "NOTE: no DART-Eval data in $DART_DIR (dart_download.py): 'other' skips s5"
fi

LOG_DIR="$BENCH_REPO/benchmarks/results/logs"
mkdir -p "$LOG_DIR"
JOB="$BENCH_REPO/benchmarks/sherlock/job.sh"
TAG="gkm${QUICK:+q}"
# Quick runs are small: short limits, and the default GPU job memory (128 GB+
# can wait a day for a node with that much free)
[[ -n $QUICK ]] && for k in "${!TIME[@]}"; do TIME[$k]=04:00:00; done && MEM=()

submit() { # name, sbatch options..., -- job.sh arguments
    local name=$1 opts=() id
    shift
    while [[ $1 != -- ]]; do opts+=("$1"); shift; done
    shift
    local cmd=(sbatch --parsable --job-name="$TAG-$name" --output="$LOG_DIR/%x-%j.out"
        --kill-on-invalid-dep=yes "${opts[@]}" "$JOB" "$@")
    if [[ $DRY -eq 1 ]]; then
        echo "${cmd[*]}" >&2
        echo "DRY-$name"
    else
        id=$("${cmd[@]}")
        echo "$name: $id ($*)" >&2
        echo "$id"
    fi
}
cpu_opts() { # unit
    echo --partition="$CPU_PARTITION" --constraint="$CPU_CONSTRAINT" --exclusive \
        --mem=0 --time="${TIME[$1]}"
}
gpu_opts() { # unit
    echo --partition="$GPU_PARTITION" --constraint="$GPU_CONSTRAINT" --gpus=1 \
        --cpus-per-task="$GPU_CPUS" --mem="${MEM[$1]:-64G}" --time="${TIME[$1]}"
}

echo "Repo $BENCH_REPO @ $(cd "$BENCH_REPO" && git rev-parse --short HEAD)"
if [[ -n "$(cd "$BENCH_REPO" && git status --porcelain --untracked-files=no)" ]]; then
    echo "WARNING: uncommitted changes; results will be tagged <commit>-dirty"
fi
echo "CPU: $CPU_PARTITION [$CPU_CONSTRAINT], whole node; GPU ($GPU): $GPU_PARTITION" \
    "[$GPU_CONSTRAINT], $GPU_CPUS CPUs; $THREADS threads; panels: ${PANELS[*]}"

setup_dep=()
if [[ $SETUP -eq 1 ]]; then
    want_ctcf=
    [[ " ${PANELS[*]} " == *" s3b "* ]] && want_ctcf=s3b
    id=$(submit setup --partition="$PARTITION" --cpus-per-task=8 --mem=32G \
        --time=06:00:00 -- setup $want_ctcf)
    setup_dep=(--dependency=afterok:$id)
fi

gpu_prev=
gpu_dep() { # setup afterok, plus afterany on the previous GPU job
    local d=${setup_dep[0]:-}
    [[ -n $gpu_prev ]] && d=${d:+$d,}afterany:$gpu_prev
    [[ -n $d ]] && d=${d/#--dependency=/} && echo "--dependency=$d"
}
for p in "${PANELS[@]}"; do
    if [[ $p == other ]]; then
        # shellcheck disable=SC2046
        gpu_prev=$(submit other $(gpu_opts other) $(gpu_dep) -- other $OTHER_TARGETS)
        continue
    fi
    [[ -n ${CPU_UNITS[$p]:-} ]] || { echo "Unknown panel: $p"; exit 1; }
    parts=()
    for u in ${CPU_UNITS[$p]}; do
        # shellcheck disable=SC2046
        parts+=("$(submit "$u" $(cpu_opts "$u") ${setup_dep[@]+"${setup_dep[@]}"} -- cpu "$u")")
    done
    for u in ${GPU_UNITS[$p]}; do
        # shellcheck disable=SC2046
        gpu_prev=$(submit "$u" $(gpu_opts "$u") $(gpu_dep) -- gpu "$u")
        parts+=("$gpu_prev")
    done
    dep=$(IFS=:; echo "afterok:${parts[*]}")
    submit "$p-merge" --partition="$PARTITION" --cpus-per-task=1 --mem=8G \
        --time=00:30:00 --dependency="$dep" -- merge "$p" >/dev/null
done
echo "Logs: $LOG_DIR. Check with: squeue --me"
