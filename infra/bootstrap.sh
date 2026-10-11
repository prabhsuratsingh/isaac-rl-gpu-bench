#!/usr/bin/env bash
# Idempotent instance bootstrap. Invoked by the Vast onstart hook (see launcher/instance.py) from
# /workspace/repo after the repo is cloned at $GIT_SHA. Safe to re-run (e.g. after a container restart).
#
# Result: /workspace/.ready exists  <=>  Isaac Sim boots headless and all step-1 tasks step on the GPU.
# Failure: /workspace/.bootstrap_failed contains the failing step; the full log is /workspace/bootstrap.log.
set -euo pipefail

WS=/workspace
REPO=$WS/repo
STAMP=$WS/.bootstrap_stamp
cd "$REPO"

step() { echo; echo "===== [bootstrap $(date -u +%H:%M:%S)] $* ====="; CURRENT_STEP="$*"; }
trap 'echo "FAILED at: ${CURRENT_STEP:-?}" | tee $WS/.bootstrap_failed' ERR
rm -f $WS/.ready $WS/.bootstrap_failed

# Env vars passed to the container are visible to onstart but not to later ssh sessions.
# Persist the ones we need so jobs launched over ssh see the same environment.
step "persist environment"
# Keep a previously stamped GPUBENCH_DPH (written by the launcher) across re-runs.
OLD_DPH=$(grep '^export GPUBENCH_DPH=' /etc/gpubench.env 2>/dev/null || true)
# NVIDIA_VISIBLE_DEVICES is deliberately not persisted: Vast sets it to "void" inside the container.
env | grep -E '^(GPUBENCH_|GIT_SHA|GH_REPO|OMNI_KIT_ACCEPT_EULA|ACCEPT_EULA|PRIVACY_CONSENT|NVIDIA_DRIVER_CAPABILITIES|CONTAINER_ID|VAST_)' \
  | sed 's/^/export /; s/=\(.*\)$/="\1"/' > /etc/gpubench.env || true
echo 'export PYTHONUNBUFFERED=1' >> /etc/gpubench.env
[ -n "$OLD_DPH" ] && ! grep -q '^export GPUBENCH_DPH=' /etc/gpubench.env && echo "$OLD_DPH" >> /etc/gpubench.env || true
grep -q gpubench.env /root/.bashrc 2>/dev/null || echo '[ -f /etc/gpubench.env ] && . /etc/gpubench.env' >> /root/.bashrc

step "apt packages (headless Kit runtime libs, tools)"
export DEBIAN_FRONTEND=noninteractive
rm -f /etc/apt/sources.list.d/nvidia-devtools.list  # stale unsigned repo from an older bootstrap version
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  git curl wget ca-certificates gnupg tmux htop pciutils build-essential \
  libglu1-mesa libxrender1 libxt6 libxi6 libsm6 libice6 libxrandr2 libxinerama1 libxcursor1 \
  libxkbcommon0 libegl1 libgl1 libvulkan1 vulkan-tools >/dev/null

# The NVIDIA container toolkit mounts the Vulkan driver when NVIDIA_DRIVER_CAPABILITIES includes
# graphics, but some hosts ship no ICD manifest. Kit needs it even headless. Exactly one manifest must
# exist: two make Vulkan enumerate the same GPU twice.
FALLBACK_ICD=/usr/share/vulkan/icd.d/nvidia_icd.json
if [ -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
  if [ -f $FALLBACK_ICD ] && grep -q '"api_version": "1.3" } }' $FALLBACK_ICD; then rm -f $FALLBACK_ICD; echo "removed duplicate fallback ICD"; fi
elif [ ! -f $FALLBACK_ICD ]; then
  mkdir -p /usr/share/vulkan/icd.d
  cat > $FALLBACK_ICD <<'EOF'
{ "file_format_version": "1.0.0", "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version": "1.3" } }
EOF
  echo "wrote fallback nvidia_icd.json"
fi
vulkaninfo --summary 2>/dev/null | grep -E 'deviceName' || echo "WARNING: vulkaninfo found no device"

step "Nsight Systems CLI"
# From the CUDA apt repo (signed by cuda-keyring). Not fatal: only profiling jobs need it.
if ! command -v nsys >/dev/null 2>&1; then
  # (explicit && chain: `set -e` is ignored inside a `( ... ) || ...` list)
  {
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/ck.deb \
      && dpkg -i /tmp/ck.deb >/dev/null \
      && apt-get update -qq \
      && apt-get install -y -qq --no-install-recommends cuda-nsight-systems-12-8 >/dev/null \
      && NSYS_BIN=$(find /opt/nvidia /usr/local/cuda* -type f -name nsys -path '*/bin/*' 2>/dev/null | sort -V | tail -1) \
      && [ -n "$NSYS_BIN" ] && ln -sf "$NSYS_BIN" /usr/local/bin/nsys
  } || echo "WARNING: nsys install failed; profiling jobs will fail but training is unaffected"
fi
command -v nsys >/dev/null && nsys --version || true

step "python packages"
# Snapshot of what the base image ships, taken once before we install anything. `gpubench-vast freeze`
# subtracts it so the lock only contains what we add (base packages are pinned by the image itself).
[ -f $WS/base_freeze.txt ] || python -m pip list --format=freeze > $WS/base_freeze.txt
python -m pip install -q --upgrade pip
REQ=requirements/remote.lock.txt
[ -s "$REQ" ] || { echo "no lock file yet -> using loose pins (run 'gpubench-vast freeze' after this)"; REQ=requirements/remote.in.txt; }
HASH=$(cat "$REQ" pyproject.toml | sha256sum | cut -d' ' -f1)
if [ -f "$STAMP" ] && [ "$(cat $STAMP)" = "$HASH" ]; then
  echo "requirements unchanged since last bootstrap; skipping pip install"
else
  python -m pip install -q -r "$REQ" --extra-index-url https://pypi.nvidia.com
  echo "$HASH" > "$STAMP"
fi
python -m pip install -q -e ".[remote]" --no-deps
python -m pip install -q numpy pandas pytest >/dev/null  # no-ops when already satisfied by the lock

step "sanity: C compiler (Triton builds its launcher stub; torch.compile needs it too)"
gcc --version | head -1

step "sanity: torch sees the GPU"
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch cannot see a GPU"
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
EOF

step "smoke test (Isaac Sim headless + step every task)"
python scripts/smoke_test.py --out $WS/smoke.json

touch $WS/.ready
echo "BOOTSTRAP OK"
