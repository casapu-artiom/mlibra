#!/usr/bin/env python
# encoding: utf-8

from typing import Optional

import torch

from .riemann_kernel import RiemannKernel
from ..operators import PrecisionMaternOperator


class RiemannMaternKernel(RiemannKernel):
    has_lengthscale = True

    def __init__(
        self,
        nu: Optional[int] = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nu = nu
        # When learning free per-mode weights (see RiemannKernel), warm-start them
        # AT the Matern density computed with the current lengthscale/diffusion_scale,
        # so optimisation begins exactly at the tied kernel and can only improve.
        # NB: if the experiment sets `lengthscale` AFTER construction, this reflects
        # the pre-set lengthscale — a harmless starting point for a learnable param.
        if getattr(self, "learn_spectral_weights", False):
            with torch.no_grad():
                self._set_spectral_weights(
                    self._matern_density().reshape(*self.batch_shape, 1, self.num_modes)
                )

    def _matern_density(self):
        # The tied, monotone (isotropic-in-lambda) Matern spectral density.
        # diffusion_scale (init 1.0, frozen unless learn_diffusion_scale=True) is a
        # multiplicative scale on the frozen spectrum: lambda_k -> diffusion_scale*lambda_k.
        # It needs no eigenpair recompute (scaling an operator leaves eigvecs unchanged),
        # and is the multiplicative companion to the lengthscale's additive 2*nu/l^2 floor.
        # Co-locate eigval with the (learnable) kernel params: during the
        # learn_spectral_weights warm-start in __init__ the params are still on CPU while
        # eigval may already be on the compute device (the kernel's .to(device) runs AFTER
        # construction). In the forward path they are already co-located, so .to is a no-op.
        ls = self.lengthscale
        safe_eigval = self.eigval.clamp(min=0.0).to(ls.device)
        return (2*self.nu / ls.square()
                + self.diffusion_scale * safe_eigval).pow(-self.nu)

    def spectral_density(self):
        # Free per-mode weights (anisotropic in lambda) when enabled — these can
        # up-weight the specific modes that discriminate a region boundary, which a
        # monotone Matern density cannot. Otherwise the tied Matern density.
        if getattr(self, "learn_spectral_weights", False):
            return self.spectral_weights
        return self._matern_density()

    def precision(self):
        return PrecisionMaternOperator(self.laplacian(), self.nu, self.lengthscale)
