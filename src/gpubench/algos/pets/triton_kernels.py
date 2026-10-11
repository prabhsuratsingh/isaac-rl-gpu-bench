"""Hand-written Triton kernels for the fused PETS planner (step-3 ablation level `fused`).

pets_step_kernel: ONE launch per horizon step, replacing every elementwise op that follows the ensemble's
last batched GEMM in the naive planner (about 15-25 small kernels, x E members in the naive version):

    raw head [E*N, 2*DO]  ->  soft-bounded log-variance  ->  Gaussian sample (in-kernel Philox RNG)
    -> de-normalised delta -> next state (+ action-slice overwrite) -> task reward and termination
    -> alive mask and return accumulation
    -> next model input written in place (state | next action), so no torch.cat per step.

sample_actions_kernel: one launch per MPPI iteration: noise * sigma + mean, clamp, first sequence = mean.

RNG: the seed is read from a device tensor, so a CUDA graph replay picks up the incremented seed and
draws fresh noise; offsets are passed per (iteration, step). For equivalence tests, both kernels can
read externally supplied standard-normal noise instead (EXT_NOISE).

Rows are particles in the planner's member-major layout: row = member * N + particle.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

TASKS = {"cartpole": 0, "ant": 1}


@triton.jit
def _softplus(x):
    # matches torch.nn.functional.softplus(beta=1, threshold=20)
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))


@triton.jit
def pets_step_kernel(
    head_ptr, maxlv_ptr, minlv_ptr, omean_ptr, ostd_ptr,
    s_in_ptr, s_out_ptr, x_out_ptr, act_ptr, act_next_ptr,
    noise_ptr, seed_ptr, ret_ptr, alive_ptr,
    rows, N, rng_offset, stride_act, stride_act_next,
    c0, c1, c2, c3, c4, c5, c6, c7,
    DO: tl.constexpr, A: tl.constexpr, DO_P: tl.constexpr, A_P: tl.constexpr,
    ACT_LO: tl.constexpr, TASK: tl.constexpr, SAMPLE: tl.constexpr, EXT_NOISE: tl.constexpr,
    HAS_NEXT: tl.constexpr, BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    rm = r < rows
    c = tl.arange(0, DO_P)
    cm = c < DO
    m2 = rm[:, None] & cm[None, :]
    e = r // N
    r64 = r.to(tl.int64)

    # ---- model head -> sampled, de-normalised next state ------------------------------------------
    mean = tl.load(head_ptr + r64[:, None] * (2 * DO) + c[None, :], mask=m2, other=0.0).to(tl.float32)
    lvr = tl.load(head_ptr + r64[:, None] * (2 * DO) + DO + c[None, :], mask=m2, other=0.0).to(tl.float32)
    mx = tl.load(maxlv_ptr + e[:, None] * DO + c[None, :], mask=m2, other=0.0)
    mn = tl.load(minlv_ptr + e[:, None] * DO + c[None, :], mask=m2, other=0.0)
    lv = mx - _softplus(mx - lvr)
    lv = mn + _softplus(lv - mn)
    if SAMPLE:
        if EXT_NOISE:
            eps = tl.load(noise_ptr + r64[:, None] * DO + c[None, :], mask=m2, other=0.0)
        else:
            seed = tl.load(seed_ptr)
            eps = tl.randn(seed, rng_offset + r[:, None] * DO + c[None, :])
        dn = mean + eps * tl.exp(0.5 * lv)
    else:
        dn = mean
    om = tl.load(omean_ptr + c, mask=cm, other=0.0)
    osd = tl.load(ostd_ptr + c, mask=cm, other=1.0)
    s = tl.load(s_in_ptr + r64[:, None] * DO + c[None, :], mask=m2, other=0.0)
    nxt = s + dn * osd[None, :] + om[None, :]
    if ACT_LO >= 0:  # observation contains the previous action: known exactly
        ac = c - ACT_LO
        amask = m2 & (ac[None, :] >= 0) & (ac[None, :] < A)
        av = tl.load(act_ptr + r64[:, None] * stride_act + ac[None, :], mask=amask, other=0.0)
        nxt = tl.where(amask, av, nxt)
    nxt = tl.where(m2, nxt, 0.0)
    tl.store(s_out_ptr + r64[:, None] * DO + c[None, :], nxt, mask=m2)
    tl.store(x_out_ptr + r64[:, None] * (DO + A) + c[None, :], nxt, mask=m2)  # cast to model dtype
    if HAS_NEXT:
        ca = tl.arange(0, A_P)
        ma = rm[:, None] & (ca[None, :] < A)
        an = tl.load(act_next_ptr + r64[:, None] * stride_act_next + ca[None, :], mask=ma, other=0.0)
        tl.store(x_out_ptr + r64[:, None] * (DO + A) + DO + ca[None, :], an, mask=ma)

    # ---- task reward / termination on the predicted state (envs/costs.py, same formulas) -------------
    if TASK == 0:  # cartpole: c0 alive, c1 term, c2 pole_pos, c3 cart_vel, c4 pole_vel, c5 max_cart
        pole = tl.sum(tl.where(c[None, :] == 0, nxt, 0.0), axis=1)
        pole_v = tl.sum(tl.where(c[None, :] == 1, nxt, 0.0), axis=1)
        cart = tl.sum(tl.where(c[None, :] == 2, nxt, 0.0), axis=1)
        cart_v = tl.sum(tl.where(c[None, :] == 3, nxt, 0.0), axis=1)
        term = (tl.abs(cart) > c5) | (tl.abs(pole) > 1.5707963267948966)
        t = term.to(tl.float32)
        rew = c0 * (1.0 - t) + c1 * t + c2 * pole * pole + c3 * tl.abs(cart_v) + c4 * tl.abs(pole_v)
    else:  # ant: c0 heading_w, c1 up_w, c2 act_cost, c3 energy, c4 alive, c5 death, c6 term_h, c7 decimation
        z = tl.sum(tl.where(c[None, :] == 0, nxt, 0.0), axis=1)
        vx = tl.sum(tl.where(c[None, :] == 1, nxt, 0.0), axis=1)
        vy = tl.sum(tl.where(c[None, :] == 2, nxt, 0.0), axis=1)
        ang = tl.sum(tl.where(c[None, :] == 9, nxt, 0.0), axis=1)
        up = tl.sum(tl.where(c[None, :] == 10, nxt, 0.0), axis=1)
        hd = tl.sum(tl.where(c[None, :] == 11, nxt, 0.0), axis=1)
        progress = c7 * (vx * tl.cos(ang) + vy * tl.sin(ang))
        heading_r = tl.where(hd > 0.8, c0, c0 * hd / 0.8)
        up_r = tl.where(up > 0.93, c1, 0.0)
        ca = tl.arange(0, A_P)
        ma = rm[:, None] & (ca[None, :] < A)
        a_t = tl.load(act_ptr + r64[:, None] * stride_act + ca[None, :], mask=ma, other=0.0)
        actions_cost = tl.sum(a_t * a_t, axis=1)
        jv = c - 20  # dof_vel columns 20..27 pair with action j = c - 20
        mv = m2 & (jv[None, :] >= 0) & (jv[None, :] < A)
        a_v = tl.load(act_ptr + r64[:, None] * stride_act + jv[None, :], mask=mv, other=0.0)
        electricity = tl.sum(tl.where(mv, tl.abs(a_v * nxt), 0.0), axis=1)
        lim = (c[None, :] >= 12) & (c[None, :] < 20) & (nxt > 0.98)
        at_limit = tl.sum(tl.where(lim & m2, 1.0, 0.0), axis=1)
        rew = progress + c4 + up_r + heading_r - c2 * actions_cost - c3 * electricity - at_limit
        term = z < c6
        rew = tl.where(term, c5, rew)
        t = term.to(tl.float32)

    alive = tl.load(alive_ptr + r, mask=rm, other=0.0)
    ret = tl.load(ret_ptr + r, mask=rm, other=0.0)
    tl.store(ret_ptr + r, ret + alive * rew, mask=rm)
    tl.store(alive_ptr + r, alive * (1.0 - t), mask=rm)


@triton.jit
def sample_actions_kernel(
    mean_ptr, sigma_ptr, noise_ptr, seed_ptr, out_ptr,
    total, K, HA, rng_offset, lo, hi,
    SIGMA_PER_ELEM: tl.constexpr, EXT_NOISE: tl.constexpr, BLOCK: tl.constexpr,
):
    """out[b, k, j] = clamp(mean[b, j] + sigma * eps[b, k, j], lo, hi), with eps[:, 0] = 0 (the current mean)."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < total
    b = idx // (K * HA)
    k = (idx // HA) % K
    j = idx % HA
    mu = tl.load(mean_ptr + b * HA + j, mask=m, other=0.0)
    if SIGMA_PER_ELEM:
        sig = tl.load(sigma_ptr + b * HA + j, mask=m, other=0.0)
    else:
        sig = tl.load(sigma_ptr)
    if EXT_NOISE:
        eps = tl.load(noise_ptr + idx, mask=m, other=0.0)
    else:
        seed = tl.load(seed_ptr)
        eps = tl.randn(seed, rng_offset + idx)
    eps = tl.where(k == 0, 0.0, eps)
    v = mu + eps * sig
    v = tl.minimum(tl.maximum(v, lo), hi)
    tl.store(out_ptr + idx, v, mask=m)


def _p2(x: int) -> int:
    return 1 << (max(1, x) - 1).bit_length()


def launch_step(head, max_lv, min_lv, out_mean, out_std, s_in, s_out, x_out, act_t, act_next, ret, alive,
                seed, rng_offset: int, task: str, params, act_lo: int, sample: bool = True,
                noise: torch.Tensor | None = None, block_r: int = 32):
    """All tensors [E, N, *] contiguous except act_t/act_next (row-strided views of [E, N, H, A])."""
    E, N, DO = s_in.shape
    A = act_t.shape[-1]
    rows = E * N
    p = list(params) + [0.0] * (8 - len(params))
    grid = (triton.cdiv(rows, block_r),)
    pets_step_kernel[grid](
        head, max_lv, min_lv, out_mean, out_std,
        s_in, s_out, x_out, act_t, act_next if act_next is not None else act_t,
        noise if noise is not None else s_in, seed, ret, alive,
        rows, N, rng_offset, act_t.stride(1), (act_next if act_next is not None else act_t).stride(1),
        *[float(v) for v in p],
        DO=DO, A=A, DO_P=_p2(DO), A_P=_p2(A), ACT_LO=act_lo, TASK=TASKS[task], SAMPLE=sample,
        EXT_NOISE=noise is not None, HAS_NEXT=act_next is not None, BLOCK_R=block_r,
    )


def launch_sample(mean, sigma, seed, out, rng_offset: int, lo: float, hi: float, noise: torch.Tensor | None = None,
                  block: int = 1024):
    """mean [B, Hc, A]; sigma 0-d tensor or [B, 1, Hc, A]; out [B, K, Hc, A] contiguous."""
    B, K, Hc, A = out.shape
    total = out.numel()
    sample_actions_kernel[(triton.cdiv(total, block),)](
        mean, sigma, noise if noise is not None else out, seed, out,
        total, K, Hc * A, rng_offset, float(lo), float(hi),
        SIGMA_PER_ELEM=sigma.dim() > 0, EXT_NOISE=noise is not None, BLOCK=block,
    )
