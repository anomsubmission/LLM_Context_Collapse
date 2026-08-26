# `runner/` — the infrastructure-agnostic wrapper

Nothing in this repository's science depends on a cloud provider, a scheduler, or a
particular GPU. This directory is the thin layer that lets you spread the work over
whatever hardware you actually have.

The contract is deliberately small. The repository promises two things:

| Script | Contract |
|---|---|
| `scripts/run_shard.sh <i> <n>` | Runs shard `i` of `n` of the full grid. Writes results under `$RESULTS_DIR`. Exits non-zero on failure. Safe to re-run: completed cells are skipped. |
| `scripts/collect.sh` | Merges every shard's results into `$RESULTS_DIR`, de-duplicating by cell identity, then regenerates the paper's numbers and figures. |

Any launcher that can call those two things is a valid backend. Four are provided.

## Backends

```bash
# 1. local — run every shard sequentially on this machine (or set -j for parallel)
runner/launch.sh --backend local --shards 8

# 2. docker — same, inside the pinned image; one container per shard
runner/launch.sh --backend docker --shards 8 --gpus all

# 3. slurm — submit a job array; one array task per shard
runner/launch.sh --backend slurm --shards 40 --partition gpu --gpus-per-task 1

# 4. ssh — a hand-listed fleet of machines you already have (any cloud, or none)
echo -e "gpu-a.example.com\ngpu-b.example.com" > runner/hosts.txt
runner/launch.sh --backend ssh --hosts runner/hosts.txt --shards 40
```

`--shards n` is the only knob that changes the partition. The partition is
deterministic and content-addressed: shard `i` of `n` always covers the same cells,
every cell is covered exactly once, and shards never coordinate. So you can stop a
run, change `n`, and restart without double-counting — `scripts/collect.sh`
de-duplicates by cell identity, not by arrival order.

## Storage: `runner/storage.py`

Results and checkpoints are addressed by URI, not by filesystem path. Set
`RESULTS_DIR` to anything [`fsspec`](https://filesystem-spec.readthedocs.io)
understands and the code does not change:

```bash
export RESULTS_DIR=./results                      # local disk (default)
export RESULTS_DIR=s3://my-bucket/run-2026-08     # AWS S3, or any S3-compatible
export RESULTS_DIR=gs://my-bucket/run-2026-08     # Google Cloud Storage
export RESULTS_DIR=az://my-container/run          # Azure Blob
export RESULTS_DIR=file:///mnt/shared/run         # a shared NFS mount
```

Credentials are read from the provider's own standard mechanism — the environment,
`~/.aws/credentials`, a workload identity, an instance role. This repository never
reads, stores, or forwards a credential, and there is no provider-specific code path:
S3 works because `fsspec` has an S3 driver, not because anything here knows about S3.
MinIO, Ceph, and R2 work through the same S3 driver with `AWS_ENDPOINT_URL` set.

```bash
# smoke-test that your storage URI is reachable and writable before a long run:
python runner/storage.py --check "$RESULTS_DIR"
```

## Reproducing on one machine, no wrapper at all

The wrapper is optional. This is always valid:

```bash
scripts/run_shard.sh 0 1     # the entire grid, serially, right here
scripts/collect.sh
```

That is the reference path, and it is what the numbers in `EXPECTED.md` were checked
against.
