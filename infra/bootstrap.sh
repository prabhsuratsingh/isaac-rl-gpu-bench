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
env | grep -E '^(GPUBENCH_|GIT_SHA|GH_REPO|OMNI_KIT_ACCEPT_EULA|ACCEPT_EULA|PRIVACY_CONSENT|NVIDIA_|CONTAINER_ID|VAST_)' \
  | grep -v '^GH_TOKEN' | sed 's/^/export /; s/=\(.*\)$/="\1"/' > /etc/gpubench.env || true
echo 'export PYTHONUNBUFFERED=1' >> /etc/gpubench.env
grep -q gpubench.env /root/.bashrc 2>/dev/null || echo '[ -f /etc/gpubench.env ] && . /etc/gpubench.env' >> /root/.bashrc

step "apt packages (headless Kit runtime libs, tools)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
  git curl wget ca-certificates gnupg tmux htop pciutils \
  libglu1-mesa libxrender1 libxt6 libxi6 libsm6 libice6 libxrandr2 libxinerama1 libxcursor1 \
  libxkbcommon0 libegl1 libgl1 libvulkan1 vulkan-tools >/dev/null

# The NVIDIA container toolkit mounts the Vulkan driver when NVIDIA_DRIVER_CAPABILITIES includes
# graphics, but some hosts ship no ICD manifest. Kit needs it even headless.
if ! ls /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json >/dev/null 2>&1; then
  mkdir -p /usr/share/vulkan/icd.d
  cat > /usr/share/vulkan/icd.d/nvidia_icd.json <<'EOF'
{ "file_format_version": "1.0.0", "ICD": { "library_path": "libGLX_nvidia.so.0", "api_version": "1.3" } }
EOF
  echo "wrote fallback nvidia_icd.json"
fi

step "Nsight Systems CLI"
if ! command -v nsys >/dev/null 2>&1; then
  (
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/ck.deb \
      && dpkg -i /tmp/ck.deb >/dev/null \
      && echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /" \
         > /etc/apt/sources.list.d/nvidia-devtools.list \
      && apt-get update -qq && apt-get install -y -qq --no-install-recommends nsight-systems-cli >/dev/null
  ) || echo "WARNING: nsys install failed; profiling jobs will fail but training is unaffected"
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
