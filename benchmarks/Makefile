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

PY            ?= python
DEVICE        ?= cuda
LSGKM_ENV     ?= lsgkm
LSGKM_BIN     ?=
DART_WORK_DIR ?= data/dart-eval
DART_MODELS   ?= data/models
DART_ENCODE   ?= ENCFF953DDE ENCFF701NCI

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
DART_ARGS     := --max-variants 2000
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
DART_ARGS     :=
endif

ifeq ($(DEVICE),cpu)
SWEEP_DEVICE  := cpu
else
SWEEP_DEVICE  := both
endif

COMMON := _common.py
# gkmexplain may be missing from conda's ls-gkm; point LSGKM_BIN at a source build
LSGKM  := --conda-env $(LSGKM_ENV) $(if $(LSGKM_BIN),--lsgkm-bin $(LSGKM_BIN))
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
	$(PY) timing_train.py --conda-env $(LSGKM_ENV) --device $(DEVICE) \
	    --sizes $(TIMING_SIZES) --output $@

$(OUT)/timing_train_T1.tsv: timing_train.py $(COMMON)
	$(PY) timing_train.py --conda-env $(LSGKM_ENV) --device $(DEVICE) \
	    --sizes $(TIMING_SIZES) --threads 1 --output $@

$(OUT)/timing_variants.tsv: timing_variants.py $(DSQTL)
	$(PY) timing_variants.py --device $(DEVICE) $(LSGKM) $(VARIANT_ARGS) --output $@

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
S3 := $(OUT)/training_memory.tsv
S4 := $(OUT)/completeness.tsv
S5 := $(foreach m,$(DART_ENCODE),$(OUT)/dart_task4_$(m).tsv $(OUT)/dart_task5_dsqtl_$(m).tsv)

# S1 is about training, so never reuse cached models
$(OUT)/dsqtl_retrained_per_model.tsv: $(DSQTL)
	$(PY) dsqtl_auprc.py --device $(DEVICE) --params $(RETRAINED) $(DSQTL_QUICK) --force-train \
	    --output $(OUT)/dsqtl_retrained_summed.tsv --per-model-output $@

$(OUT)/throughput.tsv: throughput_sweep.py $(COMMON)
	$(PY) throughput_sweep.py --device $(SWEEP_DEVICE) --output $@

$(OUT)/training_memory.tsv: training_memory.py $(DSQTL)
	$(PY) training_memory.py --device $(DEVICE) $(LSGKM) $(SOLVER_ARGS) --output $@

$(OUT)/completeness.tsv: completeness.py $(COMMON)
	$(PY) completeness.py --device $(DEVICE) $(COMPL_ARGS) --output $@

$(OUT)/dart_task4_%.tsv: dart_eval.py $(COMMON) $(DART_MODELS)/%.model.txt.gz
	$(PY) dart_eval.py -m $(DART_MODELS)/$*.model.txt.gz --task activity \
	    --cell-lines GM12878 --device $(DEVICE) --work-dir $(DART_WORK_DIR) \
	    --output $@ -v

$(OUT)/dart_task5_dsqtl_%.tsv: dart_eval.py $(COMMON) $(DART_MODELS)/%.model.txt.gz
	$(PY) dart_eval.py -m $(DART_MODELS)/$*.model.txt.gz --task vep \
	    --vep-tasks dsqtl --vep-method gkmexplain --device $(DEVICE) \
	    --work-dir $(DART_WORK_DIR) $(DART_ARGS) --output $@ -v

$(DART_MODELS)/%.model.txt.gz:
	mkdir -p $(DART_MODELS)
	curl -fL -o $@ https://www.encodeproject.org/files/$*/@@download/$*.txt.gz

s1: $(S1)
s2: $(S2)
s3: $(S3)
s4: $(S4)
s5: $(S5)

# ── Aggregates ──────────────────────────────────────────────────────

paper: fig1a fig1b fig1c fig1d s1 s2 s3 s4 s5

quick:
	$(MAKE) paper QUICK=1

figures:
	RESULTS_DIR=$(OUT) FIGURES_DIR=$(FIG_OUT) $(PY) -m jupyter nbconvert \
	    --to notebook --execute --inplace figures.ipynb

help:
	@echo "Figure 1:   fig1a fig1b fig1c fig1d"
	@echo "Supplement: s1 s2 s3 s4 s5"
	@echo "All:        paper (results), figures (PDFs), quick (small run)"
	@echo "Options:    DEVICE=$(DEVICE) QUICK=1 PY=$(PY) LSGKM_ENV=$(LSGKM_ENV) LSGKM_BIN=<dir>"
	@echo "            DART_WORK_DIR=$(DART_WORK_DIR) DART_MODELS=$(DART_MODELS)"

.DEFAULT_GOAL := help
.PHONY: fig1a fig1b fig1c fig1d s1 s2 s3 s4 s5 paper quick figures help
.SECONDARY:
