import torch

from npi_gnfc.config import Config
from npi_gnfc.geometry import synthetic_geometry
from npi_gnfc.train import build_model, make_loss_weights


def test_forward_smoke():
    cfg = Config()
    cfg.model.eeg_channels = 8
    cfg.model.n_regions = 8
    cfg.model.n_eigenmodes = 4
    cfg.model.latent_dim = 8
    cfg.model.text_embedding_dim = 16
    cfg.model.hidden_dim = 16
    cfg.model.diffusion_steps = 5
    geometry = synthetic_geometry(n_vertices=10, n_eigenmodes=4)
    model = build_model(cfg, geometry, torch.device("cpu"))
    eeg = torch.randn(2, 8, 20)
    y = torch.randn(2, 16)
    out = model(eeg, y, loss_weights=make_loss_weights(cfg))
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    assert out.encoder.field.shape == (2, 20, 10)
