# Benchmark protocol

**Status:** v0.1 (step 1). The sections marked **FROZEN** may not change after the step-1 PPO runs. Any change
after that point re-runs every affected method.

The question is when a learned model beats massively parallel simulation. This document fixes what
"beats" means, how it is measured and how every method is given a fair chance.

---

## 1. Tasks (FROZEN)

| alias      | gym id                                   | obs | act | control dt | episode   | model-based cost        |
|------------|------------------------------------------|-----|-----|------------|-----------|-------------------------|
| `cartpole` | `Isaac-Cartpole-Direct-v0`               | 4   | 1   | 1/60 s     | 300 steps | exact port              |
| `ant`      | `Isaac-Ant-Direct-v0`                    | 36  | 8   | 1/60 s     | 900 steps | approx. port (progress) |
| `anymal`   | `Isaac-Velocity-Flat-Anymal-C-Direct-v0` | 48  | 12  | 1/50 s     | 1000 steps| not ported (PPO only in step 1) |

- **Software stack:** Isaac Lab 2.3.2.post1, Isaac Sim 5.1 (PhysX GPU), torch 2.7.0+cu128, rsl-rl-lib 3.0.1 (pinned by the isaaclab wheel) and Python 3.11. The exact package set is `requirements/remote.lock.txt` on top of the pinned base image in `infra/gpus.yaml`.
- **Env config:** the Isaac Lab defaults for every task, plus only the overrides listed in `configs/task/*.yaml`.
- **Cost functions:** model-based methods plan with the torch reward/termination functions in `src/gpubench/envs/costs.py`. They are validated against the simulator's own rewards by `scripts/check_costs.py`, and the results are stored in `results/<gpu>/checks/`. Cartpole must match within 1e-4. For Ant we report the correlation and the mean absolute error.
- **Sim-cost knob (crossover):** `task.sim_substep_mult = k` divides `sim.dt` by k and multiplies `decimation` by k. Control rate, episode length and the obs/action spaces are unchanged, and each env step costs about k× the physics.

## 2. Methods and implementations

| method | implementation | notes |
|---|---|---|
| PPO | `reference`: rsl_rl `OnPolicyRunner` with Isaac Lab's shipped agent config | 4096 envs by default; the 512→8192 sweep comes later |
| PETS | `naive`: plain PyTorch, a Python loop over ensemble members and horizon, fp32 | Axis-D baseline |
| PETS | `fused` (later): batched ensemble, fused sampling/cost, CUDA graphs, bf16 | one optimisation at a time |
| MBPO | `naive`: SAC (2×256, auto-entropy) trained on 5% real + 95% branched model data. It uses PETS's ensemble, training routine and known reward/termination functions (the model learns dynamics only). TS-1 rollouts; GPU replay for real and model data. | step 2 |
| random | uniform actions; the floor of the "learns" criterion | step 2 |
| Dreamer | later steps (nice-to-have) | |

**GPU-native requirement:** in our own code, planning, model inference, model training and replay run entirely on the GPU. No `.cpu()`, `.item()` or boolean-mask indexing happens inside the act loop. This is checked by a `sync_debug=error` run (`torch.cuda.set_sync_debug_mode`) in `configs/exp/step1_pets.yaml`, which passed in step 1. That torch mode is a prototype that does not detect every synchronising op, so it is evidence rather than proof; the Nsight trace (no `cudaStreamSynchronize`/`cudaMemcpy` DtoH inside `plan`) is the second check. Syncs inside the simulator's `step()` (for example Isaac Lab's reset bookkeeping) are attributed to the `sim` phase, and are the same for every method.

**Auto-reset masking:** Isaac Lab resets finished envs inside `step()`, so the returned `next_obs` of a done env is the post-reset observation. Model-based learners keep such transitions in the buffer with `valid=0`, and they are excluded from model training by loss weighting (no sync). PPO uses rsl_rl's standard handling (time-out bootstrapping as configured by Isaac Lab).

## 3. Seeds and statistics (FROZEN)

- **Evaluation seeds:** 0, 1, 2, 3, 4 for every (method, task, GPU) cell. The seed sets torch/numpy/python RNGs, `env_cfg.seed` and the agent seed.
- **Tuning seeds:** 100, 101, 102. They are never used for reported numbers.
- **Aggregation:** mean ± 95% CI from a percentile bootstrap over seeds (10,000 resamples). With n=5 we also show every seed as a thin line or dot, and never report a mean alone.
- **Determinism:** runs use `deterministic=False` (the fast kernels), TF32 is on (the Isaac Lab default), and cuDNN benchmark is off. Across GPUs, "identical" means the same seeds, lock file and configs. Bitwise-identical trajectories are not expected.

## 4. Metrics

| metric | definition |
|---|---|
| env step | one transition of one env (`num_envs` transitions per vectorised `step()`) |
| `env_steps_per_s` | env transitions per wall-clock second over the log interval, with everything included (sim, planning, learning) |
| `model_rollouts_per_s` | model transitions (particle × horizon step) evaluated per second of planner time |
| `planner_steps_per_s` | real-env actions produced per second of planner time (`num_envs` × plan calls / plan time) |
| `reward_mean` | mean undiscounted return of the **last min(100, num_envs) finished training episodes**: rsl_rl's 100-episode rewbuffer for PPO (4096 envs), an on-GPU ring in `GpuVecEnv` for model-based methods (32 envs, so 32 episodes). The rule gives every method about the same lag, one episode duration. A fixed 100-episode window would span about 3 generations of episodes at 32 envs; in `step1b_pets` it still held the random-action episodes at the end of training and reported about 120 while the policy scored 250-298. Returns come from the behaviour policy that actually collects the data. |
| **wall-clock to threshold** (headline) | `wall_s` at the first log row where `reward_mean ≥ τ_task` and the next row also satisfies it. `wall_s` starts after Isaac Sim boot, so boot is the same for all methods and is reported separately in `smoke.json`. It includes env creation. |
| samples to threshold | `env_steps` at the same row |
| GPU-hours per solved task | wall-clock to threshold / 3600 (1 GPU). Cost in $ = GPU-hours × the instance's $/hr from `run_meta.json`. |
| time breakdown | share of wall time per phase from the run's own timers (`host_ms`, figure `time_breakdown.png`), plus Nsight breakdowns (§7) for GPU busy time per phase |
| memory | `torch.cuda.max_memory_allocated/reserved` per log interval (the simulator's own allocations are not included) |

**Reward thresholds τ (FROZEN once filled in):** τ_task = 0.9 × the mean over the 5 evaluation seeds of PPO's final `reward_mean` (mean of the last 5 log rows), using the untuned reference config on the RTX 4090. Runs that never reach τ are reported as censored (> budget) and are not dropped.

**"Learns" (development gates, step 2 on):** a (method, task) learns if the 95% bootstrap CI lower bound of its final return is above the uniform-random policy's return (`algo=random`, ≥256 episodes, 3 seeds; `results/<gpu>/step2_random`), **and** its mean final return is at least 10% of PPO's final return (cartpole ≥ 29.6, ant ≥ 1,667). This criterion only gates development. τ-based wall-clock stays the headline metric, and the gap to τ is always reported.

| task | PPO final reward (mean ± CI) | τ |
|---|---|---|
| cartpole | 295.89 [295.72, 296.06] | **266.30** |
| ant | 16,670.48 [16,074.10, 17,250.17] | **15,003.43** |
| anymal | 25.94 [25.65, 26.23] | **23.35** |

Frozen 2026-10-06 in `configs/thresholds.yaml` from `results/rtx4090/step1_ppo` (git `64de724`); reproduce with `python scripts/analyze.py results/rtx4090 --derive-thresholds`.

## 5. Tuning budget (the fairness rule)

Every method gets the same hyperparameter search budget per task:

- **The budget is wall-clock, and only wall-clock (v0.5).** Every trial and every development baseline gets the same per-run cap: 300 s on cartpole, 1200 s on ant. Step and iteration limits are set to effectively unlimited (`algo.total_env_steps=1e8`, raised `max_iterations`) so that they can never end a run first.
  - Why this rule exists: in round 1 of the step-2 tuning, PETS and MBPO carried fixed step budgets (40k / 300k) that ended 9 of 16 cartpole trials after 85–150 s of their 300 s, and every PETS-ant trial after 475–1,017 s of its 1,200 s. Meanwhile PPO had its iteration limit raised. Round 2 (`tune_r2_*`) re-ran the same sampled configurations under this rule.
  - Round 1's spend is still reported, as "discarded GPU-h".
- **Search:** seeded random search (`scripts/tune.py`, no external dependency). The budget is in `configs/tuning/budget.yaml` and is identical for every method. A re-run round re-tests the same configurations, because the sampler seed does not depend on the round.
  - **Step 2 (small budget):** 8 trials per (method, task), 1 tuning seed (100), with per-trial wall-clock caps of 300 s (cartpole) and 1200 s (ant).
  - **Later (full budget):** 20 trials × 3 tuning seeds.
  - Trial 0 is always the untuned default.
  - Trials stop as soon as they hold τ (`run.stop_at_tau`), which saves budget without changing the objective.
- **Objective (lower is better):** wall-clock seconds until the run holds τ. A trial that never does scores `cap × (2 − clip(final/τ, 0, 1))`, so it ranks behind every solved trial but still by how close it got.
- **Search spaces:** declared in `configs/tuning/<method>.yaml` before any tuning run, with a similar number of dimensions for every method (about 6). PPO's space includes `num_envs`, learning rate, `num_steps_per_env`, epochs, minibatches and entropy coefficient.
- **Accounting:** the GPU-hours and dollars spent tuning are logged per method and task (`results/<gpu>/analysis/tuning_budget.{md,tex}`) and reported alongside the results.
- **Use:** the best trial becomes `configs/tuned/<algo>_<task>.yaml` and is applied with `tuned=true` (run label `impl+tuned`).
- **Order of reporting:** the untuned reference numbers (Isaac Lab defaults for PPO, the literature defaults in `configs/algo/*.yaml` for model-based methods) are reported *in addition to* the tuned ones.

## 6. Logging format

Each run writes `results/<exp>/<run_id>/`, and the launcher pulls it to `results/<gpu>/<exp>/<run_id>/`:

- `run_meta.json`
  - schema version
  - run id, algo, impl, task, seed, exp, start time
  - `sys`: GPU model, driver, max CUDA supported by the driver, SM count, clocks, torch CUDA/cuDNN, package versions, git SHA
  - `vast`: GPU alias, offer id, $/hr, image
- `config.json`: the fully resolved config.
- `agent_cfg.json`: the resolved rsl_rl agent config (PPO only).
- `metrics.jsonl`: one row per log interval. Every algorithm writes it through the shared `gpubench.common.reporter.Reporter`, so the common keys are identical by construction and checked by a test. Algorithm-specific values are prefixed `<algo>/` (e.g. `pets/model_holdout_mse`, `mbpo/q_loss`); the step-1 files use the unprefixed names.
  ```json
  {"run_id", "algo", "impl", "task", "seed", "wall_s", "iteration", "env_steps",
   "env_steps_per_s", "model_rollouts_per_s", "planner_steps_per_s",
   "reward_mean", "ep_len_mean", "episodes",
   "gpu_ms": {"sim":…, "plan":…, "model_train":…, "policy_act":…, "policy_update":…, "buffer":…},
   "host_ms": {…same keys…}, "calls": {…},
   "mem_alloc_max_mb", "mem_reserved_max_mb",
   "model_loss", "model_train_mse", "model_holdout_mse",      // PETS
   "collection_s", "learn_s", "loss_*"}                        // PPO (rsl_rl's own timers/losses)
  ```
- `summary.json`: status (`ok` or `failed` with a traceback), total wall time, GPU-hours, cost, env creation time, physics and control dt.

`gpu_ms` is measured between CUDA events on the current stream. It equals device time for GPU-bound phases, but for host-bound phases such as a naive Python loop, `host_ms` is the honest number. Both are kept, and the figures state which one they use.

## 7. Nsight Systems recipe

```bash
bash scripts/profile.sh <name> algo=<a> task=<t> seed=0 profile.start=<i0> profile.length=<n>
```

- The run records `--trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi`. The window is opened and closed from Python after warmup, so boot and JIT are excluded.
- Windows: PPO uses iterations 20–50. PETS uses env iterations 150–350, after the first model fit.
- Phases are NVTX ranges with the same names as the `gpu_ms` keys. `scripts/nsys_breakdown.py` reads the exported `.sqlite` and reports, per phase, host time and **GPU busy time**: the summed duration of kernels/memcpy/memset whose launching runtime call happened while that phase was open (on any thread, so PhysX's worker-thread launches count as `sim`). nsys' own `nvtx_gpu_proj_sum` spans first-to-last GPU op and counts idle gaps, so it is kept only for reference. `_total.gpu_util` = union of device activity / capture window.
- Tracing inflates host time for launch-heavy code (about 3x for naive PETS), so profiles are used for proportions and GPU-busy time, never for absolute wall-clock.
- Profiles are taken from separate runs and are never mixed into the timing results, because profiler overhead would contaminate them.

## 8. Hardware protocol (Axis C)

- Hardware: one GPU per instance, with the GPU class defined by `infra/gpus.yaml`. Hosts must be verified, have reliability > 0.98 and a driver supporting CUDA ≥ 12.8.
- Software: the same lock file, base image and commit on every GPU. The launcher refuses dirty or unpushed trees.
- Each instance runs `scripts/smoke_test.py` before any job, and its report (`smoke.json`: boot time and random-action env-steps/s per task) is archived with the results.
- Only one job runs on a GPU at a time.
- **One comparison = one host.** Every method in a comparison table runs on the same instance, so the GPU *and* the host CPU are the same, and PPO is re-run next to each new method. Reason: launch-bound code is CPU-sensitive. On two RTX 4090 hosts, naive PETS planning took 157 ms/call on a Threadripper 2950X against 71 ms on an EPYC 9684X (2.2×); model training was 2.0× slower and simulation 1.4× slower. The host CPU is recorded in every `run_meta.json` (`sys.cpu`). `analyze.py` warns when a task's runs mix CPUs, and `gpubench-vast offers/up/exp --cpu REGEX` pins the CPU class.
- **Cross-host reproducibility (measured):** with the same seed, lock file and GPU model, the PETS Cartpole learning signal was bit-identical across two hosts with different CPUs and drivers (570.86 vs 595.71): 95/95 log rows identical over 5 seeds.
- Known risk: A100 and H100 have no RT cores, and Isaac Sim officially targets RTX GPUs. Headless PhysX support is checked with `gpubench-vast smoke a100` before we plan runs there.

## 9. Change log

- v0.1: initial protocol (step 1).
- v0.9.3 (2026-10-17), scope: MBPO/PETS on Anymal-C are out of scope.
  - **Why:** the reward needs unobserved torques, accelerations, air time and contacts. A port means either a lossy cost (a different task) or a learned reward head (new algorithm code after the freeze), at about 6–10 GPU-h. Anymal's 20.2 ms natural sim step is Ant-equivalent k ≈ 3.6, far below k\* ≈ 18.
  - **Kept:** Anymal is used only as a measured sim-cost reference on the crossover axis (PPO step-1 runs, same host timer).
- v0.9.2 (2026-10-15), bug fix:
  - **Bug:** Isaac Lab's Ant reward computes progress as Δ(−distance to target / `cfg.sim.dt`). The substep knob divides `sim.dt` by k, so the progress term, and with it the task, was multiplied by k. Our Ant cost function scaled the same way through `decimation`.
    - At k ≥ 4, PPO reached τ in about 4,500 vector steps instead of about 25,000.
    - MBPO reached the full τ at every k ≥ 4, which it never does at k = 1.
  - **Consequence:** the first Ant crossover run (`step5_crossover_ant`) is **invalid** and is kept on disk only as a record.
  - **Fix:** `envs/isaac.py::_undo_dt_scaled_progress` removes (1 − 1/k) of the progress term on non-terminated envs. This is exact given Isaac Lab's step order: `_get_dones` updates the potentials, then `_get_rewards` runs, and terminated envs receive `death_cost`. `costs.py` uses decimation / k.
  - **Verification:** CPU tests (`tests/test_substep_knob.py`) and on-GPU `check_costs --substep-mult {1,4,16}`, run at the start of `step5_crossover_ant_fixed`.
  - **Unaffected:** Cartpole's reward has no dt term; its samples-to-τ is flat across k (F16), so the Cartpole crossover stands. k = 1 runs are unaffected.
  - The v0.9.1 prediction (0.3 τ crossing near k ≈ 14; none at 0.5 τ) carries over unchanged to the rerun.
- v0.9.1 (2026-10-15), before any Ant crossover run: the Ant grid is k in {1, 4, 8, 16, 32} (k = 2 replaced by 32).
  - **Why:** the Cartpole crossover showed that wall-clock to tau ~ (sequential vector steps to tau) x (sim ms per step + method overhead per step). The sim step costs the same at N = 32 and N = 4096 (latency-bound).
  - **Prediction from the k = 1 Ant runs:** PPO needs about 5,000 vector steps to 0.3 tau and MBPO fast about 3,600 (overheads about 4.5 and 42 ms per step beyond an 8 ms sim step), so the 0.3-tau crossing should land near k ~ 15. At 0.5 tau (about 8,700 against 10,000 steps), there should be no crossing.
  - **Change:** k = 32 brackets that crossing; k = 2 carries little information.
  - Recorded as a prediction before the runs.
- v0.9 (2026-10-12), step 5 (pre-registered before any step-5 run):
  - **Code:** step-5 runs use the freeze-v1 algorithm code. Every later commit only adds analysis/plotting scripts and experiment configs; there are no algorithm changes.
  - **Simulator cost knob:** `task.sim_substep_mult = k` divides `sim.dt` by k and multiplies `decimation` by k. The control step (and so the task, the episode length and τ) is unchanged; each environment step does k× the physics substeps. Finer physics can shift returns slightly, so τ stays fixed from the k = 1 PPO reference and PPO's final return per k is reported as a check.
  - **Crossover:**
    - For each task and model-based method, k\* is the smallest k at which its median wall-clock to τ beats the best PPO (over N) at the same k.
    - It is found by linear interpolation of log(t_PPO / t_MB) against log2 k between bracketing grid points; censored seeds count as +∞ in the median.
    - **Ant thresholds:** no model-based method reaches τ on Ant, so Ant is scored at **0.3 τ and 0.5 τ**, fixed now.
  - **Early stopping:** sweep runs stop once they hold τ (`run.stop_at_tau`, same rule as the analysis); the metric is wall-clock to τ.
  - **Grids:**
    - **Ablation:** PETS-Cartpole, levels naive → batched → fused → graph → bf16 → fast, 3 seeds.
    - **Scaling:**
      - PPO N ∈ {256, 1024, 4096, 16384} on Cartpole, {1024, 4096, 16384} on Ant;
      - PETS-fast E ∈ {1, 2, 4, 5, 8, 10}. E must divide samples × particles = 200, so 3 and 7 are excluded;
      - (K, H) ∈ {(100, 36), (200, 36), (400, 36), (800, 36), (200, 12), (200, 72)}.
    - **Crossover:**
      - Cartpole: k ∈ {1, 2, 4, 8, 16} × PPO N ∈ {1024, 4096, 16384} + PETS fast + MBPO fast;
      - Ant: k ∈ {1, 2, 4, 8, 16} × PPO N = 4096 + MBPO fast.
    - 3 seeds everywhere. The k = 1 points come from the main benchmark and the scaling sweep.
  - **Cross-GPU:** A100 and H100, seeds 0–2.
    - Cartpole: PPO default and PETS/MBPO naive and fast.
    - Ant: PPO default and MBPO fast.
    - Each GPU has its own host CPU (the CPU cannot be matched across GPU types). It is recorded and reported alongside the results.
- v0.8 (2026-10-12), step 4:
  - MBPO `fast` = batched ensemble + CUDA-graphed model training + fast rollouts (TS-1 by random permutation, i.e. each member takes an equal random chunk of particles per step; same marginal member distribution, sampled without replacement) with the PETS Triton step and bf16 GEMMs, captured as CUDA graphs + CUDA-graphed SAC updates (capturable Adam, in-graph minibatch sampling).
  - PETS `fast` = the step-3 bf16 graphed planner + CUDA-graphed model training.
  - **Code freeze:** after the step-4 smoke passes, the code is tagged `freeze-v1`. The main benchmark runs at that tag; afterwards only bug fixes, each recorded here with its commit.
  - **Dreamer is cut:** no RSSM exists, so by the gate-4 rule there is no profiling panel either.
  - **Main benchmark (`main_cartpole`, `main_ant`):** PPO default + tuned; PETS and MBPO naive + fast with tuned hyperparameters; 5 seeds; wall-clock budget 300 / 1200 s.
  - **Gate 4:** every must-have cell has 5 completed seeds.
  - **Deviation (2026-10-12):** the `main_ant` session hit `--max-hours` (30 runs x 1200 s is over 10 h of compute, not the 6.5 h I budgeted) during `mbpo_fast_ant` seed 2. That partial run was discarded. Seeds 2-4 were rerun at the frozen code by `main_ant_rerun` on a second EPYC 9684X host (the same CPU model). No code changed.
- v0.7.1 (2026-10-11): throughput rates (`planner_steps_per_s` etc.) use max(host, GPU-event) phase time. Host time alone overstated CUDA-graph levels, whose replay returns to Python ~6 ms before the GPU finishes; old rows are rescaled in analyze.py. The step-3 return check uses a two-sample bootstrap CI of the mean difference vs naive, and a level is degraded only if that CI excludes 0. Nsight breakdowns count CUDA-graph replays (GRAPH_TRACE).
- v0.7 (2026-10-11): step-3 systems ablation for PETS, selected by `algo.impl` with identical hyperparameters (the tuned naive PETS-Cartpole config):
  - The levels are naive -> batched -> fused (hand-written Triton step kernel: log-variance bounding, in-kernel Philox sampling, de-normalisation, task reward/termination, return accumulation, next-input write; first-layer normalisation folded into the weights) -> graph (whole plan() as CUDA graphs, one per action-repeat phase) -> bf16 (planning GEMMs in bf16). FP8 is probed in the microbenchmark only.
  - **Equivalence:** levels must agree on identical external noise (`tests/test_gpu.py`; fp32 within 2e-4 on actions, bf16 within 1% on returns). End-to-end runs use a **fixed data budget** (40k env steps, no wall cap), so learning curves are comparable.
  - **Gate 3:** at least 3x naive planner throughput (median `planner_steps_per_s`) at a level whose final-return CI still reaches naive's mean.
  - The fused/graph levels draw their noise in-kernel, so they are statistically, not bitwise, equal to the torch levels.
- v0.6 result (2026-10-10): batched PETS gets 2.4x more Ant data in the same budget (1.39M vs 0.59M steps) but the same return (1,020 vs 1,070); its model error floor does not move. PETS on locomotion is dropped for good; MBPO is the locomotion contender. The batched ensemble becomes level 1 of the step-3 ablation.
- v0.6 (2026-10-09): PETS retry on Ant (step 2c). Adds the batched ensemble (`impl: batched`, one baddbmm per layer for all members), holdout early stopping for model training (its scalar checks sync only inside the model_train phase, never in the act loop), and an optional CEM-style elite refit in MPPI. Its tuning round (8 trials, same caps) is recorded as PETS-only extra development effort.
- v0.5 (2026-10-08), step-2 gate decision: on locomotion the model-based contender may be MBPO instead of PETS (mirror of the original "drop MBPO if it struggles" rule). Result: Cartpole all three learn; Ant: PPO and MBPO learn, PETS reaches 1,070 against the 1,667 bar, limited by naive speed. PETS stays on Cartpole and is re-tested on Ant after the systems optimisations (same wall budget, more steps/s).
- v0.5 (2026-10-08): the budget is wall-clock only (step/iteration limits must not bind); tuning rounds (`tune_r2_*`) with all-round spend accounting; default model-based baselines re-run as `step2b_baselines`.
- v0.4 (2026-10-07): "learns" criterion; equal-budget tuning procedure (seeded random search, objective, stop-at-τ, accounting); shared Reporter schema; MBPO and random baseline added.
- v0.3 (2026-10-06): reward window = min(100, num_envs); one comparison = one host (CPU recorded, `--cpu` filter); cross-host reproducibility measured.
- v0.2 (2026-10-06): τ frozen; `reward_mean` unified to last-100 episodes for all methods; Nsight GPU-busy attribution; base image digest guard (`infra/gpus.yaml: image_digest`).
