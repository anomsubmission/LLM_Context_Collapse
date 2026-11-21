# Characterizing and mitigating the in-context cliff

This repository contains the current paper, supplement, measured evidence and CPU
reproduction code. The analysis parses 53,200 final evaluations and checks 658
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

This release replaces obsolete public drivers and drafts with the validated
reproduction tree. Previous releases remain in Git history; original frozen model
execution records and adapters remain in the research archive. This is reanalysis
of saved measurements, not a fresh training/inference run. Some ancillary figures
retain vector artwork because query-level source records are unavailable.

Llama-3.2 regional and hosted DeepSeek-V3.2 zero-shot measurements are unavailable
and are not silently substituted. Private reviewer correspondence and author
comment threads are not part of this public code release.
