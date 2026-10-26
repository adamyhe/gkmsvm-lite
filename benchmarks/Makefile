# Application-note figures: one target per panel.
#
#   make -C benchmarks help
#   make -C benchmarks fig1d DEVICE=cuda
#   make -C benchmarks paper              # every result file
#   make -C benchmarks figures            # results/ -> figures/ (no GPU needed)
#   make -C benchmarks quick              # small settings, writes results/quick/
#
# A target is skipped when its result file is newer than its scripts, so a
# rerun after a fix only recomputes the affected panels. Force one with
# `make -B <target>` or by deleting its file.

# Per-host settings (LSGKM_BIN, THREADS, DEVICE, ...); not committed
-include local.mk

PY            ?= python
DEVICE        ?= cuda
LSGKM_ENV     ?= lsgkm
LSGKM_BIN     ?=
# mLS-GKM source build's bin directory: adds it as a baseline in 1B, 1C and S2
MLSGKM_BIN    ?=
# CPU threads for LS-GKM -T and gkmsvm-lite (1, 4 or 16); empty = largest
# the host's usable cores allow (16 on a 16+ core server)
THREADS       ?=
DART_WORK_DIR ?= data/dart-eval
DART_MODELS   ?= data/models
# Task 4: each cell line with the ENCODE gkm-SVM model trained on the DNase-seq
# experiment DART-Eval uses (cell_model). Task 5: GM12878 models, DNase and ATAC
DART_TASK4    ?= GM12878_ENCFF701NCI H1ESC_ENCFF888QKW HEPG2_ENCFF984SKX \
                 IMR90_ENCFF077GEF K562_ENCFF795HQE
DART_TASK5    ?= ENCFF701NCI ENCFF953DDE
# Models retrained on DART-Eval's training chromosomes (dart_train.py): Task 4 per
# cell line in S5; Task 5 (GM12878) a check of its own, s5-retrained
DART_RETRAIN  ?= GM12878 H1ESC HEPG2 IMR90 K562
DART_RETRAIN5 ?= GM12878
# Label for a second host's runs (e.g. SET=a100): results go to results/<SET>/ and
# figures to figures/<SET>/; panels a set doesn't have are read from results/
SET           ?=

ifdef QUICK
ROOT          := results/quick
FIG_ROOT      := figures/quick
TIMING_SIZES  := 500,1000,2000
DSQTL_QUICK   := --n-negsets 1
AGREE_ARGS    := --negsets 1 --tags t3
RETRAINED     := l10k6
COMPL_ARGS    := --n-random 20
VARIANT_ARGS  := --n-variants 500
SOLVER_ARGS   := --sizes 500,1000,2000 --n-test 200
SOLVER_CPU_ARGS := --sizes 500,1000,2000 --n-test 200
CTCF_ARGS     := --sizes 2000,5000 --n-test 200
SWEEP_ARGS    := --sizes 1000,4000 --n-seqs 200
DART_ARGS     := --max-variants 200
DART4_ARGS    := --max-seqs 2000
DART_TRAIN_ARGS := --n-pos 1000 --n-neg 1000
else
ROOT          := results
FIG_ROOT      := figures
TIMING_SIZES  := 500,1000,2000,5000,10000,20000
DSQTL_QUICK   :=
AGREE_ARGS    :=
RETRAINED     := l10k6,l10k6_rbf
COMPL_ARGS    :=
VARIANT_ARGS  :=
SOLVER_ARGS   :=
# CPU SMO with a 2,048-column cache: ~2 min at N=10k on an M4
SOLVER_CPU_ARGS := --sizes 1000,2000,5000,10000,20000
# LS-GKM's scaling study (Lee 2016, Fig. 1): its eight sizes (5k-30k in 5k
# steps, 60k, 90k) plus 45k and 75k, where its 8 GB cache fills (~60k) and the
# Gram reaches 60 GB; gkmtrain with its default 100 MB and an 8 GB kernel cache
CTCF_ARGS     := --sizes 5000,10000,15000,20000,25000,30000,45000,60000,75000,90000 \
                 --lsgkm-caches 100,8000 --timeout 172800
SWEEP_ARGS    :=
DART_ARGS     :=
DART4_ARGS    :=
DART_TRAIN_ARGS :=
endif

OUT    := $(ROOT)$(if $(SET),/$(SET))
FIG_OUT := $(FIG_ROOT)$(if $(SET),/$(SET))
COMMON := _common.py
# SPLIT=1 (benchmarks/sherlock/): each timing panel's CPU runs (LS-GKM,
# mLS-GKM, gkmsvm-lite CPU) and GPU runs are separate part files in $(PARTS),
# made in jobs on fixed CPU and GPU hardware, then merged into $(OUT). A part
# is only made in a job whose BENCH_HW matches it, so a stale or missing part
# never runs on the wrong machine.
PARTS  := $(OUT)/parts
on      = $(if $(filter $(1),$(BENCH_HW)),,$(error $@ is a $(1) part, but \
          BENCH_HW='$(BENCH_HW)': make it in a $(1) job (sherlock/submit.sh)))
THR    := $(if $(THREADS),--threads $(THREADS))
# gkmexplain is not in conda's ls-gkm: point LSGKM_BIN at a kundajelab/lsgkm source build
LSGKM  := --conda-env $(LSGKM_ENV) $(if $(LSGKM_BIN),--lsgkm-bin $(LSGKM_BIN))
MLS    := $(if $(MLSGKM_BIN),--mlsgkm-bin $(MLSGKM_BIN))
DSQTL  := dsqtl_auprc.py $(COMMON)

# ── Figure 1 ────────────────────────────────────────────────────────

FIG1A := $(OUT)/agreement_dsqtl.tsv
FIG1B := $(OUT)/timing_train.tsv $(OUT)/timing_train_T1.tsv
FIG1C := $(OUT)/timing_variants.tsv
FIG1D := $(OUT)/dsqtl_per_model.tsv

$(OUT)/agreement_dsqtl.tsv: agreement_lsgkm.py $(DSQTL)
	$(PY) agreement_lsgkm.py --device $(DEVICE) $(AGREE_ARGS) \
	    --output $(OUT)/agreement_checks.tsv --dump $@

ifdef SPLIT
$(PARTS)/timing_train.cpu.tsv: timing_train.py $(COMMON)
	$(call on,cpu)$(PY) timing_train.py $(LSGKM) $(MLS) --device cpu $(THR) \
	    --sizes $(TIMING_SIZES) --output $@

$(PARTS)/timing_train.gpu.tsv: timing_train.py $(COMMON)
	$(call on,gpu)$(PY) timing_train.py --device $(DEVICE) --gpu-only $(THR) \
	    --sizes $(TIMING_SIZES) --output $@

$(OUT)/timing_train.tsv: $(PARTS)/timing_train.cpu.tsv $(PARTS)/timing_train.gpu.tsv
	$(PY) timing_train.py --merge-parts $^ --output $@

# 1 thread: CPU runs only
$(OUT)/timing_train_T1.tsv: timing_train.py $(COMMON)
	$(call on,cpu)$(PY) timing_train.py $(LSGKM) $(MLS) --device cpu \
	    --sizes $(TIMING_SIZES) --threads 1 --output $@

VARIANT_ARRAYS := --arrays $(PARTS)/timing_variants.arrays
$(PARTS)/timing_variants.cpu.tsv: timing_variants.py $(DSQTL)
	$(call on,cpu)$(PY) timing_variants.py --device cpu $(LSGKM) $(MLS) $(THR) \
	    $(VARIANT_ARGS) $(VARIANT_ARRAYS) --output $@

$(PARTS)/timing_variants.gpu.tsv: timing_variants.py $(DSQTL)
	$(call on,gpu)$(PY) timing_variants.py --device $(DEVICE) --gpu-only $(THR) \
	    $(VARIANT_ARGS) $(VARIANT_ARRAYS) --output $@

$(OUT)/timing_variants.tsv: $(PARTS)/timing_variants.cpu.tsv $(PARTS)/timing_variants.gpu.tsv
	$(PY) timing_variants.py --merge-parts $^ $(VARIANT_ARRAYS) --output $@
else
$(OUT)/timing_train.tsv: timing_train.py $(COMMON)
	$(PY) timing_train.py $(LSGKM) $(MLS) --device $(DEVICE) $(THR) \
	    --sizes $(TIMING_SIZES) --output $@

$(OUT)/timing_train_T1.tsv: timing_train.py $(COMMON)
	$(PY) timing_train.py $(LSGKM) $(MLS) --device $(DEVICE) \
	    --sizes $(TIMING_SIZES) --threads 1 --output $@

$(OUT)/timing_variants.tsv: timing_variants.py $(DSQTL)
	$(PY) timing_variants.py --device $(DEVICE) $(LSGKM) $(MLS) $(THR) $(VARIANT_ARGS) \
	    --output $@
endif

$(OUT)/dsqtl_per_model.tsv: $(DSQTL)
	$(PY) dsqtl_auprc.py --device $(DEVICE) --params paper_t2,paper_t3 $(DSQTL_QUICK) \
	    --output $(OUT)/dsqtl_summed.tsv --per-model-output $@

fig1a: $(FIG1A)
fig1b: $(FIG1B)
fig1c: $(FIG1C)
fig1d: $(FIG1D)

# ── Supplement ──────────────────────────────────────────────────────

S1 := $(OUT)/dsqtl_retrained_per_model.tsv
S2 := $(OUT)/throughput.tsv
S3 := $(OUT)/training_memory.tsv $(OUT)/training_memory_cpu.tsv
S3B := $(OUT)/training_memory_ctcf.tsv
S4 := $(OUT)/completeness.tsv
S5 := $(foreach c,$(DART_TASK4),$(OUT)/dart_task4_$(c).tsv) \
      $(foreach c,$(DART_RETRAIN),$(OUT)/dart_retrained_task4_$(c).tsv) \
      $(foreach m,$(DART_TASK5),$(OUT)/dart_task5_$(m).tsv)

# S1 is about training, so never reuse cached models
$(OUT)/dsqtl_retrained_per_model.tsv: $(DSQTL)
	$(PY) dsqtl_auprc.py --device $(DEVICE) --params $(RETRAINED) $(DSQTL_QUICK) --force-train \
	    --output $(OUT)/dsqtl_retrained_summed.tsv --per-model-output $@

# H1-hESC CTCF peaks vs GC/repeat-matched hg19 windows (ctcf_data.py)
data/ctcf/train_pos.fa: ctcf_data.py
	$(PY) ctcf_data.py

ifdef SPLIT
$(PARTS)/throughput.cpu.tsv: throughput_sweep.py $(COMMON)
	$(call on,cpu)$(PY) throughput_sweep.py --device cpu $(LSGKM) $(MLS) $(THR) \
	    $(SWEEP_ARGS) --output $@

$(PARTS)/throughput.gpu.tsv: throughput_sweep.py $(COMMON)
	$(call on,gpu)$(PY) throughput_sweep.py --device $(DEVICE) --gpu-only $(THR) \
	    $(SWEEP_ARGS) --output $@

$(OUT)/throughput.tsv: $(PARTS)/throughput.cpu.tsv $(PARTS)/throughput.gpu.tsv
	$(PY) throughput_sweep.py --merge-parts $^ --output $@

# LS-GKM on the CPU node, gkmsvm-lite's solvers on the GPU; merged, as one
# run on both made it
$(PARTS)/training_memory.cpu.tsv: training_memory.py $(DSQTL)
	$(call on,cpu)$(PY) training_memory.py --device cpu --solvers lsgkm $(LSGKM) $(THR) \
	    $(SOLVER_ARGS) --arrays $(PARTS)/training_memory.arrays --output $@

$(PARTS)/training_memory.gpu.tsv: training_memory.py $(DSQTL)
	$(call on,gpu)$(PY) training_memory.py --device $(DEVICE) --no-lsgkm $(THR) \
	    $(SOLVER_ARGS) --arrays $(PARTS)/training_memory.arrays --output $@

$(OUT)/training_memory.tsv: $(PARTS)/training_memory.cpu.tsv $(PARTS)/training_memory.gpu.tsv
	$(PY) training_memory.py --merge-parts $^ --arrays $(PARTS)/training_memory.arrays \
	    --output $@

$(OUT)/training_memory_cpu.tsv: training_memory.py $(DSQTL)
	$(call on,cpu)$(PY) training_memory.py --device cpu --solvers libsvm,smo --no-lsgkm \
	    $(THR) $(SOLVER_CPU_ARGS) --output $@

$(PARTS)/training_memory_ctcf.cpu.tsv: training_memory.py data/ctcf/train_pos.fa $(DSQTL)
	$(call on,cpu)$(PY) training_memory.py --data ctcf --device cpu --solvers lsgkm \
	    $(LSGKM) $(THR) $(CTCF_ARGS) --arrays $(PARTS)/training_memory_ctcf.arrays \
	    --output $@

$(PARTS)/training_memory_ctcf.gpu.tsv: training_memory.py data/ctcf/train_pos.fa $(DSQTL)
	$(call on,gpu)$(PY) training_memory.py --data ctcf --device $(DEVICE) --no-lsgkm \
	    $(THR) $(CTCF_ARGS) --arrays $(PARTS)/training_memory_ctcf.arrays --output $@

$(OUT)/training_memory_ctcf.tsv: $(PARTS)/training_memory_ctcf.cpu.tsv \
                                 $(PARTS)/training_memory_ctcf.gpu.tsv
	$(PY) training_memory.py --merge-parts $^ \
	    --arrays $(PARTS)/training_memory_ctcf.arrays --output $@
else
$(OUT)/throughput.tsv: throughput_sweep.py $(COMMON)
	$(PY) throughput_sweep.py --device $(DEVICE) $(LSGKM) $(MLS) $(THR) $(SWEEP_ARGS) --output $@

$(OUT)/training_memory.tsv: training_memory.py $(DSQTL)
	$(PY) training_memory.py --device $(DEVICE) $(LSGKM) $(THR) $(SOLVER_ARGS) \
	    --output $@

# gkmsvm-lite solvers on CPU alone; LS-GKM rows come from the run above
$(OUT)/training_memory_cpu.tsv: training_memory.py $(DSQTL)
	$(PY) training_memory.py --device cpu --solvers libsvm,smo --no-lsgkm $(THR) \
	    $(SOLVER_CPU_ARGS) --output $@

$(OUT)/training_memory_ctcf.tsv: training_memory.py data/ctcf/train_pos.fa $(DSQTL)
	$(PY) training_memory.py --data ctcf --device $(DEVICE) $(LSGKM) $(THR) $(CTCF_ARGS) \
	    --output $@
endif

$(OUT)/completeness.tsv: completeness.py $(COMMON)
	$(PY) completeness.py --device $(DEVICE) $(COMPL_ARGS) --output $@

# dart_task4_<cell>_<model>.tsv
cell_of  = $(firstword $(subst _, ,$(1)))
model_of = $(lastword $(subst _, ,$(1)))
.SECONDEXPANSION:
$(OUT)/dart_task4_%.tsv: dart_eval.py $(COMMON) $$(DART_MODELS)/$$(call model_of,$$*).model.txt.gz
	$(PY) dart_eval.py -m $(DART_MODELS)/$(call model_of,$*).model.txt.gz --task activity \
	    --cell-lines $(call cell_of,$*) --device $(DEVICE) --work-dir $(DART_WORK_DIR) \
	    $(DART4_ARGS) --output $@ -v

$(OUT)/dart_task5_%.tsv: dart_eval.py $(COMMON) $(DART_MODELS)/%.model.txt.gz
	$(PY) dart_eval.py -m $(DART_MODELS)/$*.model.txt.gz --task vep \
	    --vep-tasks all --vep-method gkmexplain --device $(DEVICE) \
	    --work-dir $(DART_WORK_DIR) $(DART_ARGS) --output $@ -v

$(DART_MODELS)/%.model.txt.gz:
	mkdir -p $(DART_MODELS)
	curl -fL -o $@ https://www.encodeproject.org/files/$*/@@download/$*.txt.gz

# Retrained models: dart_train.py writes the model and dart_retrained_train_<cell>.tsv
S5R_MODELS := $(DART_MODELS)/retrained$(if $(QUICK),/quick)
S5R := $(foreach c,$(DART_RETRAIN5),$(OUT)/dart_retrained_task5_$(c).tsv)

$(OUT)/dart_retrained_train_%.tsv: dart_train.py dart_eval.py $(COMMON)
	$(PY) dart_train.py --cell-line $* --device $(DEVICE) --work-dir $(DART_WORK_DIR) \
	    $(DART_TRAIN_ARGS) --model-output $(S5R_MODELS)/$*.npz --output $@

$(OUT)/dart_retrained_task4_%.tsv: dart_eval.py $(COMMON) $(OUT)/dart_retrained_train_%.tsv
	$(PY) dart_eval.py -m $(S5R_MODELS)/$*.npz --task activity --cell-lines $* \
	    --device $(DEVICE) --work-dir $(DART_WORK_DIR) $(DART4_ARGS) --output $@ -v

$(OUT)/dart_retrained_task5_%.tsv: dart_eval.py $(COMMON) $(OUT)/dart_retrained_train_%.tsv
	$(PY) dart_eval.py -m $(S5R_MODELS)/$*.npz --task vep \
	    --vep-tasks all --vep-method gkmexplain --device $(DEVICE) \
	    --work-dir $(DART_WORK_DIR) $(DART_ARGS) --output $@ -v

s1: $(S1)
s2: $(S2)
s3: $(S3)
s3b: $(S3B)
s4: $(S4)
s5: $(S5)
s5-retrained: $(S5R)

# ── Parts (SPLIT=1) ─────────────────────────────────────────────────
# One target per Sherlock job: <panel>-cpu on the CPU node, <panel>-gpu on the
# GPU node; <panel> itself then only merges.

fig1b-cpu:    $(PARTS)/timing_train.cpu.tsv
fig1b-cpu-t1: $(OUT)/timing_train_T1.tsv
fig1b-gpu:    $(PARTS)/timing_train.gpu.tsv
fig1c-cpu:    $(PARTS)/timing_variants.cpu.tsv
fig1c-gpu:    $(PARTS)/timing_variants.gpu.tsv
s2-cpu:       $(PARTS)/throughput.cpu.tsv
s2-gpu:       $(PARTS)/throughput.gpu.tsv
s3-cpu:       $(PARTS)/training_memory.cpu.tsv
s3-cpu-lite:  $(OUT)/training_memory_cpu.tsv
s3-gpu:       $(PARTS)/training_memory.gpu.tsv
s3b-cpu:      $(PARTS)/training_memory_ctcf.cpu.tsv
s3b-gpu:      $(PARTS)/training_memory_ctcf.gpu.tsv

# Every download the panels need, so concurrent jobs never fetch the same file
data: $(foreach m,$(sort $(DART_TASK5) $(foreach c,$(DART_TASK4),$(call model_of,$(c)))),\
          $(DART_MODELS)/$(m).model.txt.gz)
	$(PY) -c "import dsqtl_auprc as b, timing_variants as tv; \
	    b.download_data(); b.download_sequence_sets(); \
	    [b._load_paper_model(t, i) for t in tv.PAPER_TAGS.values() for i in range(1, 6)]"

# ── Aggregates ──────────────────────────────────────────────────────

paper: fig1a fig1b fig1c fig1d s1 s2 s3 s4 s5

quick:
	$(MAKE) paper QUICK=1

figures:
	RESULTS_DIR=$(OUT) RESULTS_ROOT=$(ROOT) FIGURES_DIR=$(FIG_OUT) $(PY) -m jupyter nbconvert \
	    --to notebook --execute --inplace figures.ipynb

help:
	@echo "Figure 1:   fig1a fig1b fig1c fig1d"
	@echo "Supplement: s1 s2 s3 s4 s5; s3b (large-N CTCF scaling, not in paper: up to a day)"
	@echo "Checks:     s5-retrained (DART-Eval Task 5 with the retrained GM12878 model)"
	@echo "All:        paper (results), figures (PDFs), quick (small run), data (downloads)"
	@echo "SPLIT=1:    <panel>-cpu / <panel>-gpu parts (fig1b fig1c s2 s3 s3b; also"
	@echo "            fig1b-cpu-t1, s3-cpu-lite), then <panel> merges; see sherlock/"
	@echo "Options:    DEVICE=$(DEVICE) QUICK=1 SET=<label> PY=$(PY) THREADS=1|4|16 LSGKM_ENV=$(LSGKM_ENV) LSGKM_BIN=<dir> MLSGKM_BIN=<dir>"
	@echo "            DART_WORK_DIR=$(DART_WORK_DIR) DART_MODELS=$(DART_MODELS)"

.DEFAULT_GOAL := help
.PHONY: fig1a fig1b fig1c fig1d s1 s2 s3 s3b s4 s5 s5-retrained paper quick figures help \
        fig1b-cpu fig1b-cpu-t1 fig1b-gpu fig1c-cpu fig1c-gpu s2-cpu s2-gpu \
        s3-cpu s3-cpu-lite s3-gpu s3b-cpu s3b-gpu data
.SECONDARY:
