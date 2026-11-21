# Characterizing and mitigating the in-context cliff

The current study reports prompt/readout dependence, stage-associated output
compliance, controlled state transfer, and objective- and region-dependent
adaptation. Regional training does not locate the original failure. High-context
gains can coexist with severe zero-shot output costs; all adverse InternLM
results and null intervention results are included.

From this directory, install the analysis dependencies and reproduce the current
follow-up tables, figure and every paired contrast:

```bash
python -m pip install -e .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m publication.rebuild
python -m unittest publication.test_integrity
```

The rebuild command runs offline on CPU and includes the retained behavioral, mechanistic and objective-study generators. Add `--compile` to build the paper and supplement with Tectonic (and the response letter when present in a private source package). First-time TeX use may download public fonts/packages. The follow-up analysis verifies the input manifest, parses all
53,200 saved final outputs, recomputes raw and calibrated label predictions,
checks query/draw/training-block pairing, recalculates the bootstrap contrasts,
and compares every estimate and interval with the independent analysis. It fails
on changed inputs, missing observations or numerical disagreement. Outputs include
Figure 3, the main regional table, supplementary tables, complete contrasts,
confusion counts and `validation.json`. K=0 is one observation per query/adapter.
A confidence interval for a single-fit checkpoint does not describe training
variability. Intervals are pointwise, without multiplicity correction.

`publication/data/manifest.json` identifies the exact evidence and unavailable
measurements. The measurement export contains the original generated text, label
scores, independently reconstructed calibration priors, query IDs and source
hashes. Configurations pin model/tokenizer/data versions and evaluation settings.
`protocol_companions.json.gz` preserves the registered inputs, model pins,
pilot diagnostics and source-freeze manifests. `evidence_admission.json` records
the completed independent validation: the original G1/G2 collection-time
`scientific_evidence: false` flag is preserved, and technical pilots remain
separate from final evidence.
The three-fit regional whole-prompt comparison is separate from the four-objective
study: contextual query-label training B must not be conflated with whole-prompt A.

The submission package also includes the measured source tables and generators for
the behavioral, mechanistic and controlled-objective figures and supplement. These
studies retain their own query panels, protocols and provenance. The complete
inventory includes descriptive observations with incomplete original targeting;
they are not evidence for the verified regional cohort's targeting or a causal
failure locus. Null and adverse scientific records are never filtered by outcome.

Reanalysis of saved measurements is distinct from fresh model inference/training.
The frozen execution configurations and source snapshots accompany the project;
model payloads, licenses/access and a compatible accelerator are required for fresh
execution. Numerical identity across different hardware is not promised. Llama-3.2
regional results and the hosted DeepSeek-V3.2 zero-shot extension are unavailable.
No current command silently schedules either measurement or substitutes another
checkpoint. Local DeepSeek-7B is a different model.

The source implementation in `cliff/` supports the recorded experiments; its broad
research drivers are not the publication reproduction interface. Historical
operational queues, draft reports, caches and superseded manuscript copies are
excluded from the clean submission package. Immutable experiment and audit records
remain in the working research archive.
