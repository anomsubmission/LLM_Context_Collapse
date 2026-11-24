# Scientific experiment runtime

`sources/` contains 27 authenticated source files: prompt construction and strict
parsing, Hugging Face inference, full-label likelihood and contextual calibration,
whole-prompt/query-label LoRA training, exact saved-adapter loading, query-state
patching and reciprocal layer splicing. Each file matches its recorded execution
hash. The runtime manifest selects the required dependencies; it does not include
operational queues, credentials or private reviewer correspondence.

`registrations.tar.gz` contains the exact input/configuration files for the 38
completed G1–G6 final configurations and their 20 technical pilots. It includes
query identities/texts, demonstration contexts, calibration examples, patch donors,
length-matched stimuli and language spans. `manifest.json` hashes each source and
registered file and lists the prerequisite jobs.

From the repository's `code/` directory:

```bash
python -m pip install -e '.[experiments]'
python -m publication.experiments verify
python -m publication.experiments list
python -m publication.experiments prepare /absolute/path/to/new-replication
python -m publication.experiments smoke-imports /absolute/path/to/new-replication
python -m publication.experiments run /absolute/path/to/new-replication instruct_final --dry-run
```

These commands check and prepare files without downloading or loading models.
Preparation requires a new directory outside the publication checkout. It never
copies the published results into the new output directories.

For actual inference/training, provision the exact model and tokenizer revisions
listed by `list`/`manifest.json` under the new workspace's
`code/.cache/revision/hub/` Hugging Face cache. Respect the model access/license
requirements. Model loading is offline (`local_files_only=true`); the launcher
does not download payloads or borrow credentials from the research archive.
The recorded configurations use BF16 on Apple MPS; adequate compatible memory and
the model-specific dependencies are required. The observed software versions are
recorded in `environment.json`; the optional dependencies are installation bounds,
not a claim that all combinations reproduce the recorded numerical results.

Run the technical pilot, then its final configuration:

```bash
python -m publication.experiments run /absolute/path/to/new-replication instruct_pilot
python -m publication.experiments run /absolute/path/to/new-replication instruct_final
```

Only `run` without `--dry-run` loads model weights. Each invocation takes the
workspace's accelerator lock; a concurrent invocation fails while that lock is
held. Use the same workspace for jobs that must serialize. This lock does not
coordinate unrelated applications or other workspaces. Each final job requires
its newly completed technical pilot. Pilot checks concern complete, finite,
untruncated measurements and intervention integrity, never a desired effect sign.
For G6, the pilot performs its small registered training check; final adapter jobs
train the full registered examples. Outputs, training audits, adapter files and
record hashes are written under `revision/<campaign>/outputs/<job>/` in the new
workspace. Resume uses the original per-record provenance checks.

The portable launcher changes the workspace/lock location and constructs new
execution manifests for the shipped dependency subset. Scientific source bytes,
input bytes and model/configuration settings remain unchanged. The historical
full source freezes remain in the publication protocol companions; new manifests
are explicitly replication manifests. Fresh pilot acceptance uses the new pilot
summary, not a copied historical success record. New execution of these same
registered queries is replication, not a new independent held-out cohort.

The training library also contains the whole-prompt and query-label objectives
used in the fixed-middle study, with exact targeting, mask/EOS handling and frozen
base checks. Its interfaces are in
`sources/code/cliff/experiments/intervention.py` and
`sources/code/cliff/experiments/revision_adapter_eval.py`. The automated prepared
job catalog covers G1–G6; it does not claim to reconstruct missing historical
hosted settings or regenerate unavailable raw logit-lens observations. Older
controlled-objective results keep their separate protocol manifest and source
summaries in the paper package. Model weights and the original saved adapters
are not bundled.

All 18 regional training audits and reload checks are provided separately in
`code/publication/data/regional_training_audits.json.gz`. Exact module names,
parameter counts, seeds, update counts, calibration sequence identities and
adapter/base hashes are retained. The only redacted field is the machine-specific
adapter directory, replaced with a run-relative path; original audit hashes remain
available for comparison with the research archive. The summary
`adapter_audit.json` reports actual calibration examples, and zero-shot analysis
rows report one observed demonstration draw. Broadcasting a shared baseline for
paired contrasts does not increase its observed sample size.
