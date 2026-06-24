"""GPLFR — Gaussian Process Latent Factor Regression (GPyTorch port).

A migration of https://github.com/edstevenson/GPLFR (arXiv:2606.06576) to this
codebase. The upstream Pyro implementation uses a full N x N kernel Cholesky and
MAP latents (a low-N / high-output-dim design). Here we keep GPLFR's defining
idea — q independent latent GPs over the inputs plus a *linear-Gaussian decoder
that is marginalized analytically* ("collapsed") — but swap the full GP for the
variational inducing-point latent GP already used by ``l3di.lgp.LGP``
(``IndependentMultitaskGPModel``), so it scales to the many-pixel MALDI setting
and drops into ``maldi.experiment.MaldiExperiment`` with the same interface as
``LGP``.
"""

from .model import GPLFR

__all__ = ["GPLFR"]
