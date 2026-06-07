#!/usr/bin/env python
# encoding: utf-8

from abc import abstractmethod

import math
import torch
from torch import Tensor
import cupy as cp
import cupyx.scipy.sparse as cpsparse
import cupyx.scipy.sparse.linalg as cplinalg
import scipy.sparse as sp
from tqdm import tqdm

import gpytorch
from gpytorch.priors import Prior
from gpytorch.constraints import Positive, Interval, GreaterThan

from typing import Optional
from linear_operator.operators import LowRankRootLinearOperator, MatmulLinearOperator, RootLinearOperator

from manifold_gp.priors.inverse_gamma_prior import InverseGammaPrior

from ..utils import NearestNeighbors, bump_function
from ..operators import GraphLaplacianOperator

from torch.nn.functional import normalize

class RiemannKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True
 
    def __init__(
        self,
        *,
        # ---- Required: precomputed graph + eigenpairs ----
        knn: "NearestNeighbors",
        edge_index: torch.Tensor,
        edge_value: torch.Tensor,
        eigval: torch.Tensor,
        eigvec: torch.Tensor,
        # ---- Spectral / kernel knobs ----
        nearest_neighbors: int = 10,
        laplacian_normalization: str = "symmetric",
        num_modes: int = 100,
        bump_scale: float = 1.0,
        bump_decay: float = 0.01,
        # ---- Bandwidth ----
        graphbandwidth_init: float = 1.0,
        graphbandwidth_prior: Optional[Prior] = None,
        graphbandwidth_constraint: Optional[Interval] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
 
        # ---- Validate inputs ----
        n_nodes = knn.x.shape[0]
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"edge_index must be (2, E), got {tuple(edge_index.shape)}."
            )
        if edge_value.ndim != 1 or edge_value.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_value must be (E,) matching edge_index; got "
                f"{tuple(edge_value.shape)} vs E={edge_index.shape[1]}."
            )
        if eigvec.shape[0] != n_nodes:
            raise ValueError(
                f"eigvec has N={eigvec.shape[0]} but knn has N={n_nodes}."
            )
        if eigvec.shape[1] != eigval.shape[0]:
            raise ValueError(
                f"eigvec has {eigvec.shape[1]} columns but eigval has "
                f"{eigval.shape[0]} entries."
            )
        if eigvec.shape[1] < num_modes:
            raise ValueError(
                f"Got {eigvec.shape[1]} modes but num_modes={num_modes}. "
                f"Either pass more modes or reduce num_modes."
            )
 
        # ---- Install everything ----
        self.knn = knn
        self.edge_index = edge_index
        self.edge_value = edge_value
        # Truncate down if more modes were provided than requested.
        self.eigval = eigval[:num_modes]
        self.eigvec = eigvec[:, :num_modes]
 
        self.nearest_neighbors = nearest_neighbors
        self.laplacian_normalization = laplacian_normalization
        self.num_modes = num_modes
        self.bump_scale = bump_scale
        self.bump_decay = bump_decay
 
        # ---- Bandwidth: register learnable parameter ----
        if graphbandwidth_constraint is None:
            graphbandwidth_constraint = Positive()
 
        self.register_parameter(
            name="raw_graphbandwidth",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
        )
 
        if graphbandwidth_prior is not None:
            if not isinstance(graphbandwidth_prior, Prior):
                raise TypeError(
                    "Expected gpytorch.priors.Prior but got "
                    + type(graphbandwidth_prior).__name__
                )
            self.register_prior(
                "graphbandwidth_prior", graphbandwidth_prior,
                self._graphbandwidth_param, self._graphbandwidth_closure,
            )
 
        self.register_constraint("raw_graphbandwidth", graphbandwidth_constraint)
 
        # ---- Apply the initial bandwidth value (must match what was used
        #      for the eigensolve, since the eigvecs are tied to that value).
        self._set_graphbandwidth(torch.tensor(float(graphbandwidth_init)))
 
    # ----------------------------------------------------------------------
    # Bandwidth plumbing
    # ----------------------------------------------------------------------
    def _graphbandwidth_param(self, m) -> Tensor:
        return m.graphbandwidth
 
    def _graphbandwidth_closure(self, m, v: Tensor) -> Tensor:
        return m._set_graphbandwidth(v)
 
    def _set_graphbandwidth(self, value: Tensor):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_graphbandwidth)
        self.initialize(
            raw_graphbandwidth=self.raw_graphbandwidth_constraint.inverse_transform(value)
        )
 
    @property
    def graphbandwidth(self) -> Tensor:
        return self.raw_graphbandwidth_constraint.transform(self.raw_graphbandwidth)
 
    @graphbandwidth.setter
    def graphbandwidth(self, value: Tensor):
        self._set_graphbandwidth(value)
 
    def laplacian(self) -> GraphLaplacianOperator:
        return GraphLaplacianOperator(
            self.edge_value, self.edge_index, self.knn.x.shape[0],
            self.graphbandwidth, self.laplacian_normalization,
        )

    def _graphbandwidth_param(self, m) -> Tensor:
        # Used by the graphbandwidth_prior
        return m.graphbandwidth

    def _graphbandwidth_closure(self, m, v: Tensor) -> Tensor:
        # Used by the graphbandwidth_prior
        return m._set_graphbandwidth(v)

    def _set_graphbandwidth(self, value: Tensor):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_graphbandwidth)

        self.initialize(raw_graphbandwidth=self.raw_graphbandwidth_constraint.inverse_transform(value))


    @abstractmethod
    def spectral_density(self):
        raise NotImplementedError()

    @property
    def graphbandwidth(self) -> Tensor:
        return self.raw_graphbandwidth_constraint.transform(self.raw_graphbandwidth)

    @graphbandwidth.setter
    def graphbandwidth(self, value: Tensor):
        self._set_graphbandwidth(value)

    def laplacian(self):
        return GraphLaplacianOperator(self.edge_value, self.edge_index, self.knn.x.shape[0], self.graphbandwidth, self.laplacian_normalization)
    

    # ----------------------------------------------------------------------
    # Forward + features
    # ----------------------------------------------------------------------
    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False,
                last_dim_is_batch: bool = False, **kwargs) -> Tensor:
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)
 
        x1_eq_x2 = torch.equal(x1, x2)
        z1 = self.features(x1)
        z2 = z1 if x1_eq_x2 else self.features(x2)
 
        if diag:
            return (z1 * z2).sum(-1)
        if x1_eq_x2:
            if z1.size(-1) < z2.size(-2):
                return LowRankRootLinearOperator(z1)
            return RootLinearOperator(z1)
        return MatmulLinearOperator(z1, z2.transpose(-1, -2))

    @abstractmethod
    def spectral_density(self):
        raise NotImplementedError()
 
    def features(self, x: Tensor) -> Tensor:
        laplacian_ = self.laplacian()

        # Check if x is a *subset* of knn.x by nearest-neighbor lookup,
        # not torch.equal (which requires identical tensors).
        # For graph nodes: nearest neighbor distance should be exactly 0.
        edge_value_nn, edge_index_nn = self.knn.search(x, 1)
        is_on_graph = (edge_value_nn[:, 0] < 1e-8)   # dist² < 1e-8 → on the graph

        spectral_density = self.spectral_density()
        spectral_density = (spectral_density / spectral_density.sum().clamp(min=1e-12))
        scale = (spectral_density * self.eigvec.shape[0]).sqrt()  # (num_modes,)

        features = torch.zeros(x.shape[0], self.num_modes, device=x.device)

        # ---- In-sample rows: look up exact eigenvector by node index ----
        if is_on_graph.any():
            node_idx = edge_index_nn[is_on_graph, 0]   # which graph node
            features[is_on_graph] = scale * self.eigvec[node_idx]

        # ---- Out-of-sample rows: Nyström extension ----
        oos = ~is_on_graph
        if oos.any():
            x_oos = x[oos]
            ev, ei = edge_value_nn[oos], edge_index_nn[oos]
            # Full k-NN for Nyström (not just k=1)
            ev_k, ei_k = self.knn.search(x_oos, self.nearest_neighbors)
            within = ev_k[:, 0].sqrt() < self.bump_scale * self.graphbandwidth.squeeze()
            if within.any():
                # Use original spectral density formula for OOS
                sd_oos = self.spectral_density().div(
                    (1 - self.graphbandwidth.square() * self.eigval).square().clamp(min=1e-6)
                )
                sd_oos = (sd_oos / sd_oos.sum().clamp(min=1e-12)) * self.eigvec.shape[0]
                scale_oos = sd_oos.sqrt()
                oos_idx = oos.nonzero(as_tuple=True)[0]
                within_global = oos_idx[within]
                features[within_global] = (
                    scale_oos
                    * laplacian_.out_of_sample(
                        self.eigvec,
                        ev_k[within],
                        ei_k[within],
                    )
                    * bump_function(
                        ev_k[within, 0].sqrt(),
                        self.bump_scale * self.graphbandwidth.squeeze(),
                        self.bump_decay,
                    ).unsqueeze(-1)
                )

        return features