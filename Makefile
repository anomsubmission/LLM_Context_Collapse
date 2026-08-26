# Convenience wrappers. Everything here also works as plain `cliff <cmd>`.
PY ?= python3

.PHONY: help install install-all verify figures test smoke doctor clean reproduce

help:
	@echo "make install      tier 0/1 only (verify + figures; no torch)"
	@echo "make install-all  everything, including the experiment runners"
	@echo "make doctor       report environment and what is runnable"
	@echo "make verify       re-derive every reported number from the archive"
	@echo "make figures      regenerate Figs 1, 3, 4"
	@echo "make reproduce    verify + figures (the full no-GPU reproduction)"
	@echo "make test         run the test suite"
	@echo "make smoke        exercise the experiment pipeline with no model"

install:      ; $(PY) -m pip install -e .
install-all:  ; $(PY) -m pip install -e ".[all]"
doctor:       ; $(PY) -m cliff doctor
verify:       ; $(PY) -m cliff verify
figures:      ; $(PY) -m cliff figures
reproduce: verify figures
	@echo "\nReproduction complete: numbers verified and figures regenerated."
test:         ; $(PY) -m pytest
smoke:        ; $(PY) -m cliff sweep --backend echo --model echo --tasks ag_news --ks 0,1,4 --seeds 0 --n-test 20
clean:        ; rm -rf figures outputs .pytest_cache **/__pycache__ *.egg-info
