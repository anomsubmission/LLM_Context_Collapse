# Pinned environment for reproducing the paper.
#
# Base image choice is the one place hardware leaks into the build. The default below
# is CUDA. For CPU-only or Apple silicon, build with:
#     docker build --build-arg BASE=python:3.11-slim -f runner/Dockerfile .
# and the torch install falls back to the CPU wheel automatically.
#
# This image contains no credentials and no cloud SDK configuration. If a run writes
# to object storage, mount the host's provider config read-only at runtime -- which is
# what runner/launch.sh --backend docker does.

ARG BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep the model/dataset cache on a mountable path so it survives container churn;
    # re-downloading tens of GB of weights per shard is the single easiest way to make
    # a reproduction run cost ten times what it should.
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    MPLCONFIGDIR=/tmp/mpl

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git curl ca-certificates rsync \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /work

# Dependencies first, so editing the experiment code does not invalidate the layer
# that took ten minutes to build.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

# Optional storage drivers. Installed here so an s3:// or gs:// RESULTS_DIR works out
# of the box in the container even though the local reference run needs neither.
RUN python -m pip install "fsspec>=2024.6.0" "s3fs>=2024.6.0" "gcsfs>=2024.6.0" "adlfs>=2024.7.0"

COPY . .

RUN mkdir -p /cache && chmod -R a+rX /work
VOLUME ["/cache"]

# Fail at build time rather than after an hour of GPU time.
RUN python runner/storage.py --check /tmp/build-probe

# No ENTRYPOINT: `docker run <image> scripts/run_shard.sh 3 40` then execs exactly that
# argv, which is what runner/launch.sh relies on. The default below is the whole grid
# on one machine.
CMD ["scripts/run_shard.sh", "0", "1"]
