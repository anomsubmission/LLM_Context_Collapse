# Characterizing and mitigating the in-context cliff

This repository contains the current paper, supplement, measured evidence, CPU
reproduction code and the authenticated scientific runtime. The analysis parses 53,200 final evaluations and checks 658
estimates/confidence intervals against the independently validated analyses.
Null results, invalid outputs and adverse adaptation outcomes remain included.

```bash
cd code
python -m pip install -e .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m publication.rebuild
python -m unittest publication.test_integrity
```

Add `--compile` to rebuild the paper and supplement with Tectonic. See
[the analysis README](code/README.md) for protocols, limits and data provenance.
The current main sources are `article_cac.tex` and `supplementary_ncs.tex`.

For model execution, [the runtime guide](runtime/README.md) provides the exact
scientific source subset, registered inputs, technical pilots and isolated
execution commands for the completed G1–G6 configurations. Preparation and import
checks do not load models; actual inference/training requires an explicit `run`
command and the pinned model payloads. Full regional training/reload audits are
included with the publication data. Original saved adapter weights remain
available from the authors on request.

The CPU command above reanalyzes saved measurements. Some ancillary figures
retain vector artwork because query-level source records are unavailable.

Llama-3.2 regional and hosted DeepSeek-V3.2 zero-shot measurements are unavailable
and are not silently substituted. Private reviewer correspondence and author
comment threads are not part of this public code release.
