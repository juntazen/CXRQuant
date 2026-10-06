PYTHON ?= python3
SEEDS := 123 777 2024 31415
PRIMARY := runs/full_seed42_corrected_20260911
RUNS := $(PRIMARY) $(foreach s,$(SEEDS),runs/full_seed$(s)_20260915)
RUN ?= $(PRIMARY)
SEED ?= 42
TAG ?= local
RESUME ?=

comma := ,
empty :=
space := $(empty) $(empty)

.PHONY: test verify verify-all replay analyses aggregate divergence pycontext labellers figure preflight smoke full multiseed-queue margins data robustness revision-cpu revision-summary rebuild-corpus eosfix ablation-impression

# ---- CPU: no GPU required -------------------------------------------------------
test:
	$(PYTHON) -m pytest -q

verify:
	$(PYTHON) -m cxrquant.cli verify --results $(RUN)

verify-all:
	@for r in $(RUNS); do echo "== $$r"; $(PYTHON) -m cxrquant.cli verify --results $$r || exit 1; done

replay:
	$(PYTHON) -m cxrquant.pipeline --mode replay --run-dir $(RUN)

analyses:
	@for r in $(RUNS); do \
	  $(PYTHON) scripts/efficiency_summary.py --run-dir $$r --overwrite >/dev/null && \
	  $(PYTHON) scripts/ablation_uncertain_policy.py --run-dir $$r --overwrite >/dev/null && \
	  $(PYTHON) scripts/ablation_stratum.py --run-dir $$r --overwrite >/dev/null && echo "analyses ok: $$r" || exit 1; \
	done

aggregate:
	$(PYTHON) scripts/aggregate_seeds.py --run-dirs $(subst $(space),$(comma),$(RUNS)) --out-dir runs/multiseed_reproduced

divergence:
	@for r in $(RUNS); do $(PYTHON) scripts/divergence_analysis.py --run-dir $$r --overwrite >/dev/null && echo "divergence ok: $$r" || exit 1; done

pycontext:
	$(PYTHON) scripts/pycontext_audit.py --run-dir $(PRIMARY) --overwrite

labellers:
	$(PYTHON) scripts/baseline_evaluator_comparison.py --run-dir $(PRIMARY) --overwrite

figure:
	$(PYTHON) scripts/make_efficiency_figure.py --run-dir $(PRIMARY) --out figures_reproduced/fig_efficiency_stability.png
	$(PYTHON) scripts/make_ijies_figures.py --run-dir $(PRIMARY) --out-dir figures_reproduced

# ---- GPU: CUDA required ---------------------------------------------------------
preflight:
	$(PYTHON) -m cxrquant.pipeline --mode preflight --run-dir runs/preflight_$(TAG) --config configs/full.json

smoke:
	$(PYTHON) -m cxrquant.pipeline --mode smoke --run-dir runs/smoke_$(TAG) --config configs/smoke.json --data data/iuxray_paired.json

full:
	$(PYTHON) -m cxrquant.pipeline --mode full --run-dir runs/full_seed$(SEED)_$(TAG) \
	  --config $(if $(filter 42,$(SEED)),configs/full.json,configs/full_seed$(SEED).json) \
	  --data data/iuxray_paired.json $(if $(RESUME),--resume,)

CHECKPOINTS ?= $(PRIMARY)/checkpoints
margins:
	$(PYTHON) scripts/logit_margin_analysis.py --run-dir $(PRIMARY) --checkpoint-root $(CHECKPOINTS) --out runs/logit_margin_reproduced.json

multiseed-queue:
	$(PYTHON) scripts/multiseed_queue.py --tag $(TAG) --state audit/multiseed_queue_state.json

# ---- IJIES second-round revision (R2) ---------------------------------------------
REVOUT ?= results_local
OPENI ?= external/openi_parsed.json
OPENI_XML ?= external/openi/ecgen-radiology
OPENI_URL ?= https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz

data:                  ## CPU: download the official OpenI report release and rebuild the corpus (CC BY-NC-ND 4.0)
	mkdir -p external/openi
	test -f external/NLMCXR_reports.tgz || curl -fL --retry 5 --retry-delay 5 --retry-connrefused -o external/NLMCXR_reports.tgz $(OPENI_URL)
	@echo "8fb6de7eec73d8c3665067ad4bb003ccd57f971ae316d2642e1627ac7268667a  external/NLMCXR_reports.tgz" | sha256sum -c - \
	  || (rm -f external/NLMCXR_reports.tgz; echo "Download incomplete or corrupted; run 'make data' again."; exit 1)
	tar -xzf external/NLMCXR_reports.tgz -C external/openi
	$(PYTHON) scripts/rebuild_corpus_from_openi.py --openi $(OPENI_XML) --images data/corpus_image_ids.txt --out data/iuxray_paired.json
	$(PYTHON) scripts/parse_openi_release.py --xml-dir $(OPENI_XML) --out $(OPENI)

robustness:            ## CPU: robustness table from archived runs and revision outputs
	PYTHONPATH=src $(PYTHON) scripts/robustness_table.py --results $(REVOUT) --out $(REVOUT)/robustness.json

revision-cpu:          ## CPU: data flow, overlap, denominators/CIs, hierarchical models, efficiency, tolerances
	PYTHONPATH=src $(PYTHON) scripts/revision_analyses.py --openi $(OPENI) --out $(REVOUT)
	PYTHONPATH=src $(PYTHON) scripts/make_revision_figures.py --run-dir $(PRIMARY) --out-dir $(REVOUT)/figures

revision-summary:      ## CPU: summarise the archived revision GPU experiments (results_r2) into REVOUT
	mkdir -p $(REVOUT)
	PYTHONPATH=src $(PYTHON) scripts/summarise_reruns.py --results results_r2 --out $(REVOUT)/rerun_summary.json

rebuild-corpus:        ## CPU: rebuild data/iuxray_paired.json from the official OpenI release (CC BY-NC-ND 4.0)
	$(PYTHON) scripts/rebuild_corpus_from_openi.py --openi $(OPENI_XML) --images data/corpus_image_ids.txt \
	  --out data/iuxray_paired.rebuilt.json

eosfix:                ## GPU: regenerate all precisions of RUN with EOS exempt from the repetition penalty
	PYTHONPATH=src $(PYTHON) scripts/regenerate_checkpoints.py launch --run-dir $(RUN) --checkpoints $(CHECKPOINTS) \
	  --set repetition_penalty_exempt_eos=true --out $(REVOUT)/eosfix/$(notdir $(RUN)) --devices $(DEVICES)

ablation-impression:   ## GPU: retrain with the loss on IMPRESSION tokens only (seed 42) and audit
	$(PYTHON) -m cxrquant.pipeline --mode full --run-dir runs/ablation_impression_only_seed42 \
	  --config configs/full_seed42_impression_only.json --data data/iuxray_paired.json
