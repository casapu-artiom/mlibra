#!/usr/bin/env python
# encoding: utf-8

from typing import Optional

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

    def spectral_density(self):
        safe_eigval = self.eigval.clamp(min=0.0)
        # diffusion_scale (init 1.0, frozen unless learn_diffusion_scale=True) is a
        # multiplicative scale on the frozen spectrum: lambda_k -> diffusion_scale*lambda_k.
        # It needs no eigenpair recompute (scaling an operator leaves eigvecs unchanged),
        # and is the multiplicative companion to the lengthscale's additive 2*nu/l^2 floor.
        return (2*self.nu / self.lengthscale.square()
                + self.diffusion_scale * safe_eigval).pow(-self.nu)

    def precision(self):
        return PrecisionMaternOperator(self.laplacian(), self.nu, self.lengthscale)
