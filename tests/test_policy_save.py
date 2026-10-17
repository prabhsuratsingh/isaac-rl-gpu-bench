"""Policies saved with run.save_policy round-trip into what scripts/record.py rebuilds (CPU only)."""

import torch

from gpubench.algos.mbpo.sac import SAC
from gpubench.algos.pets.run import build_model

M = {"ensemble_size": 5, "hidden": 32, "layers": 2, "lr": 1e-3}


def test_pets_model_round_trip(tmp_path):
    model, _ = build_model(M, 4, 1, "cartpole", "cpu", "fast")
    with torch.no_grad():
        model.in_mean.normal_()  # normaliser buffers must travel with the weights
        model.in_std.uniform_(0.5, 2.0)
    torch.save({"model": model.state_dict()}, tmp_path / "policy.pt")
    clone, _ = build_model(M, 4, 1, "cartpole", "cpu", "fast")
    clone.load_state_dict(torch.load(tmp_path / "policy.pt")["model"])
    for k, v in model.state_dict().items():
        torch.testing.assert_close(clone.state_dict()[k], v)


def test_sac_actor_round_trip_with_obs_norm(tmp_path):
    sac = SAC(36, 8, hidden=32, layers=2)
    sac.set_obs_norm(torch.randn(36), torch.rand(36) + 0.5)
    torch.save({"sac": sac.state_dict()}, tmp_path / "policy.pt")
    clone = SAC(36, 8, hidden=32, layers=2)
    clone.load_state_dict(torch.load(tmp_path / "policy.pt")["sac"])
    obs = torch.randn(16, 36)
    torch.testing.assert_close(clone.act(obs, deterministic=True), sac.act(obs, deterministic=True))
