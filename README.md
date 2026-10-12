# isaac-rl-gpu-bench

**When does a learned model beat massively parallel simulation?** This repo is a wall-clock and throughput study of GPU-native model-based RL (PETS, MBPO) against massively parallel PPO on Isaac Lab. The rules for what counts as a fair comparison are in [docs/protocol.md](docs/protocol.md).

All GPU work runs on Vast.ai instances driven from your laptop. Each GPU class (4090, A100, H100, L40S, ...) is one entry in [infra/gpus.yaml](infra/gpus.yaml), so rerunning the whole study on another GPU means changing only the alias.

```
laptop (any OS)                                   Vast instance (Ubuntu 22.04, pinned image)
gpubench-vast exp rtx4090 configs/exp/x.yaml ──►  onstart: git clone @ pinned SHA
  search offers (infra/gpus.yaml)                 infra/bootstrap.sh: pip install lock, nsys, smoke test
  create instance, wait for .ready                jobrunner: jobs run one at a time, detached from ssh
  poll status                                     results/<exp>/<run_id>/{run_meta,config,metrics.jsonl,summary}
  pull results (tar over ssh)  ◄──────────────────
  destroy the instance (also on error or Ctrl-C)
```

## One-time setup (laptop)

```powershell
python -m venv .venv; .venv\Scripts\activate
pip install -e ".[launcher]"
copy .env.example .env          # set VAST_API_KEY and GPUBENCH_GH_TOKEN
```

1. **SSH key.** Run `ssh-keygen -t ed25519` if you don't already have a key, then add the `.pub` key at <https://cloud.vast.ai/manage-keys/>.
2. **GitHub.** Push this repo to a private GitHub repo. Then create a fine-grained PAT for that repo only, with **Contents: Read-only** permission, and put it in `.env` as `GPUBENCH_GH_TOKEN`.
3. **Check.** Run `gpubench-vast doctor`.

## Step 1 on the RTX 4090

```powershell
gpubench-vast offers rtx4090                       # check price and availability
gpubench-vast up rtx4090                           # first boot: about 20-30 min (Isaac Sim wheels + smoke test)
gpubench-vast freeze                               # writes requirements/remote.lock.txt: commit and push it
gpubench-vast sync                                 # instance checks out the commit that has the lock
gpubench-vast run configs/exp/step1_ppo.yaml configs/exp/step1_pets.yaml configs/exp/step1_profile.yaml
gpubench-vast status                               # job states + GPU utilisation
gpubench-vast pull                                 # results -> results/rtx4090/...
gpubench-vast down
```

Once the lock is committed, a whole experiment is a single command:

```powershell
gpubench-vast exp rtx4090 configs/exp/step1_ppo.yaml --max-hours 8
gpubench-vast smoke a100                           # does headless Isaac Sim run on this GPU? (~$1)
```

Other useful commands:

- `gpubench-vast ls` lists your live instances and what they have cost so far.
- `gpubench-vast logs --bootstrap` or `logs --job step1_ppo/ppo_ant_s0` shows the bootstrap log or one job's log.
- `gpubench-vast ssh` opens a shell on the instance.

## Step 2: PETS, MBPO, equal tuning budget, 3-seed baselines

```powershell
# all on one EPYC 9684X RTX 4090 host (one comparison = one host)
gpubench-vast exp rtx4090 configs/exp/step2_smoke.yaml --cpu "EPYC 9684X" --max-hours 1
gpubench-vast exp rtx4090 configs/exp/step2_recheck.yaml configs/exp/tune_ppo_cartpole.yaml configs/exp/tune_pets_cartpole.yaml configs/exp/tune_mbpo_cartpole.yaml `
    configs/exp/tune_ppo_ant.yaml configs/exp/tune_pets_ant.yaml configs/exp/tune_mbpo_ant.yaml `
    configs/exp/step2_baselines.yaml --cpu "EPYC 9684X" --max-hours 14
python scripts/tune.py report results/rtx4090 --paper step2     # -> configs/tuned/*.yaml + tuning budget table
git add configs/tuned; git commit -m "Tuned configs (step 2)"; git push
gpubench-vast exp rtx4090 configs/exp/step2_baselines_tuned.yaml --cpu "EPYC 9684X" --max-hours 4
python scripts/analyze.py results/rtx4090 --exps step2_random step2_baselines step2_baselines_tuned --paper step2
```

How it works:
- The tuning search spaces in `configs/tuning/` are fixed before any trial runs.
- `scripts/tune.py generate` regenerates `configs/exp/tune_*.yaml` from them.
- The gate verdict is written to `results/rtx4090/analysis/gate_step2.md`.

## Running a single job on an instance

```bash
python scripts/train.py algo=ppo task=cartpole seed=0
python scripts/train.py algo=pets_naive task=cartpole seed=0 algo.sync_debug=error
python scripts/check_costs.py --task cartpole
bash scripts/profile.sh prof_test algo=pets_naive task=cartpole profile.start=150 profile.length=200
```

Configs are `configs/algo/<algo>.yaml` + `configs/task/<task>.yaml` (including that task's `algo_overrides`) + `key=value` overrides on the command line. Experiments (`configs/exp/*.yaml`) are job matrices.

## Analysis and the research log

```powershell
pip install -e ".[analysis]"
python scripts/analyze.py results/rtx4090 --exps step1c_cartpole --paper step1c
```

This writes tables and figures to `results/rtx4090/analysis/`. With `--paper <tag>` it also copies them into `docs/paper/generated/`, where the LaTeX research log [docs/paper/findings.tex](docs/paper/findings.tex) picks them up. That log is updated with every useful finding

## Layout

| path | what |
|---|---|
| `src/gpubench/launcher/` | Vast SDK + OpenSSH launcher (`gpubench-vast`) and the remote `jobrunner` |
| `src/gpubench/common/` | phase timer (CUDA events + NVTX), JSONL logger, sysinfo, config |
| `src/gpubench/envs/` | Isaac Lab env creation, `GpuVecEnv`, torch reward/termination ports |
| `src/gpubench/algos/ppo/` | rsl_rl PPO reference with timing hooks |
| `src/gpubench/algos/pets/` | naive PETS: ensemble, MPPI, GPU replay buffer, training loop |
| `infra/` | GPU catalogue, instance bootstrap |
| `scripts/` | train / smoke test / cost check / Nsight profile + breakdown |
| `tests/` | CPU-runnable unit tests (`pytest`) |
