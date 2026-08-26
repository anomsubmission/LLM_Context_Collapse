#!/usr/bin/env bash
# Run shard i of n of the model sweep.
#
#     scripts/run_shard.sh 0 1                         # every model, one machine
#     BACKEND=hf_local scripts/run_shard.sh 2 8
#
# You do not need this to reproduce the paper: `cliff verify` re-derives all 94 reported
# numbers from the archive with no GPU. This re-runs the sweeps that produced the archive.
#
# Sharding is BY MODEL. `cliff sweep` takes --backend, so the same sweep runs against a
# local Hugging Face model, any OpenAI-compatible endpoint, or the `echo` stub:
#
#   --backend hf_local        weights from the HF Hub, in-process
#   --backend openai_compat   any /v1/chat/completions endpoint (vLLM, SGLang, a gateway)
#   --backend echo            no model at all; exercises the pipeline

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SHARD="${1:?usage: run_shard.sh <shard-index-0-based> <shard-count> [extra cliff args]}"
NSHARDS="${2:?usage: run_shard.sh <shard-index-0-based> <shard-count> [extra cliff args]}"
shift 2

BACKEND="${BACKEND:-hf_local}"
TASKS="${TASKS:-ag_news,sst2,trec}"
KS="${KS:-0,1,2,4,8,16,32,64}"
SEEDS="${SEEDS:-0,1,2}"
MODELS="${MODELS:-}"

if [ -z "$MODELS" ]; then
  # The eight models of the paper's fast-path survey, saturation-prone first.
  MODELS="meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.2-3B-Instruct \
google/gemma-3-4b-it google/gemma-3-12b-it mistralai/Mistral-7B-Instruct-v0.3 \
Qwen/Qwen2.5-7B-Instruct microsoft/Phi-3.5-mini-instruct allenai/OLMo-2-1124-7B-Instruct"
fi

echo "== pre-flight"
python3 scripts/check_env.py || true

i=0; RAN=0
for m in $MODELS; do
  if [ "$((i % NSHARDS))" -eq "$SHARD" ]; then
    echo "---- $m"
    python3 -m cliff sweep --backend "$BACKEND" --model "$m" \
      --tasks "$TASKS" --ks "$KS" --seeds "$SEEDS" "$@"
    RAN=$((RAN + 1))
  fi
  i=$((i + 1))
done

[ "$RAN" -eq 0 ] && echo "shard $SHARD of $NSHARDS had no models assigned"
echo "shard $SHARD complete. Then: scripts/collect.sh"
