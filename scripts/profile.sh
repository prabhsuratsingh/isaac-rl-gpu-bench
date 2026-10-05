#!/usr/bin/env bash
# Nsight Systems capture of a steady-state window of one training run.
#
#   bash scripts/profile.sh <name> algo=ppo task=cartpole profile.start=20 profile.length=30 [more overrides]
#
# The capture range is opened/closed from Python (gpubench.common.timing.ProfilerWindow via
# cudaProfilerStart/Stop), so Isaac Sim boot and warmup are excluded. Phases appear as NVTX ranges
# (sim / plan / model_train / policy_act / policy_update / ...). Outputs, next to the run's results:
#   results/profiles/<name>.nsys-rep
#   results/profiles/<name>_nvtx_sum.csv          host-side duration per NVTX range
#   results/profiles/<name>_nvtx_gpu_proj_sum.csv GPU work projected onto NVTX ranges
#   results/profiles/<name>_cuda_gpu_kern_sum.csv per-kernel GPU time
#   results/profiles/<name>.breakdown.json   (scripts/nsys_breakdown.py)
set -euo pipefail
NAME=$1; shift
cd "$(dirname "$0")/.."
OUT=results/profiles
mkdir -p "$OUT"
command -v nsys >/dev/null || { echo "nsys not installed (see infra/bootstrap.sh)"; exit 2; }

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-memory-usage=false \
  --force-overwrite=true \
  -o "$OUT/$NAME" \
  python scripts/train.py "$@" "run.run_id=$NAME"

nsys stats --force-export=true --report nvtx_sum --format csv --output "$OUT/$NAME" "$OUT/$NAME.nsys-rep" >/dev/null
nsys stats --report nvtx_gpu_proj_sum --format csv --output "$OUT/$NAME" "$OUT/$NAME.nsys-rep" >/dev/null || true
nsys stats --report cuda_gpu_kern_sum --format csv --output "$OUT/$NAME" "$OUT/$NAME.nsys-rep" >/dev/null || true
python scripts/nsys_breakdown.py "$OUT/$NAME" | tee "$OUT/$NAME.breakdown.json"
