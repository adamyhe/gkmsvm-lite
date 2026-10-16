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

ifdef QUICK
OUT           := results/quick
FIG_OUT       := figures/quick
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
OUT           := results
FIG_OUT       := figures
TIMING_SIZES  := 500,1000,2000,5000,10000,20000
DSQTL_QUICK   :=
AGREE_ARGS    :=
RETRAINED     := l10k6,l10k6_rbf
COMPL_ARGS    :=
VARIANT_ARGS  :=
SOLVER_ARGS   :=
# CPU SMO with a 2,048-column cache: ~2 min at N=10k on an M4
SOLVER_CPU_ARGS := --sizes 1000,2000,5000,10000,20000
# LS-GKM's scaling study (Lee 2016, Fig. 1): eight sizes from 5k to 90k, gkmtrain
# with its default 100 MB and an 8 GB kernel cache; LS-GKM can take a day at 90k
CTCF_ARGS     := --sizes 5000,10000,20000,30000,45000,60000,75000,90000 \
                 --lsgkm-caches 100,8000 --timeout 172800
SWEEP_ARGS    :=
DART_ARGS     :=
DART4_ARGS    :=
DART_TRAIN_ARGS :=
endif

COMMON := _common.py
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

$(OUT)/timing_train.tsv: timing_train.py $(COMMON)
	$(PY) timing_train.py $(LSGKM) $(MLS) --device $(DEVICE) $(THR) \
	    --sizes $(TIMING_SIZES) --output $@

$(OUT)/timing_train_T1.tsv: timing_train.py $(COMMON)
	$(PY) timing_train.py $(LSGKM) $(MLS) --device $(DEVICE) \
	    --sizes $(TIMING_SIZES) --threads 1 --output $@

$(OUT)/timing_variants.tsv: timing_variants.py $(DSQTL)
	$(PY) timing_variants.py --device $(DEVICE) $(LSGKM) $(MLS) $(THR) $(VARIANT_ARGS) \
	    --output $@

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

$(OUT)/throughput.tsv: throughput_sweep.py $(COMMON)
	$(PY) throughput_sweep.py --device $(DEVICE) $(LSGKM) $(MLS) $(THR) $(SWEEP_ARGS) --output $@

$(OUT)/training_memory.tsv: training_memory.py $(DSQTL)
	$(PY) training_memory.py --device $(DEVICE) $(LSGKM) $(THR) $(SOLVER_ARGS) \
	    --output $@

# gkmsvm-lite solvers on CPU alone; LS-GKM rows come from the run above
$(OUT)/training_memory_cpu.tsv: training_memory.py $(DSQTL)
	$(PY) training_memory.py --device cpu --solvers libsvm,smo --no-lsgkm $(THR) \
	    $(SOLVER_CPU_ARGS) --output $@

# H1-hESC CTCF peaks vs GC/repeat-matched hg19 windows (ctcf_data.py)
data/ctcf/train_pos.fa: ctcf_data.py
	$(PY) ctcf_data.py

$(OUT)/training_memory_ctcf.tsv: training_memory.py data/ctcf/train_pos.fa $(DSQTL)
	$(PY) training_memory.py --data ctcf --device $(DEVICE) $(LSGKM) $(THR) $(CTCF_ARGS) \
	    --output $@

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

# ── Aggregates ──────────────────────────────────────────────────────

paper: fig1a fig1b fig1c fig1d s1 s2 s3 s4 s5

quick:
	$(MAKE) paper QUICK=1

figures:
	RESULTS_DIR=$(OUT) FIGURES_DIR=$(FIG_OUT) $(PY) -m jupyter nbconvert \
	    --to notebook --execute --inplace figures.ipynb

help:
	@echo "Figure 1:   fig1a fig1b fig1c fig1d"
	@echo "Supplement: s1 s2 s3 s4 s5; s3b (large-N CTCF scaling, not in paper: up to a day)"
	@echo "Checks:     s5-retrained (DART-Eval Task 5 with the retrained GM12878 model)"
	@echo "All:        paper (results), figures (PDFs), quick (small run)"
	@echo "Options:    DEVICE=$(DEVICE) QUICK=1 PY=$(PY) THREADS=1|4|16 LSGKM_ENV=$(LSGKM_ENV) LSGKM_BIN=<dir> MLSGKM_BIN=<dir>"
	@echo "            DART_WORK_DIR=$(DART_WORK_DIR) DART_MODELS=$(DART_MODELS)"

.DEFAULT_GOAL := help
.PHONY: fig1a fig1b fig1c fig1d s1 s2 s3 s3b s4 s5 s5-retrained paper quick figures help
.SECONDARY:
