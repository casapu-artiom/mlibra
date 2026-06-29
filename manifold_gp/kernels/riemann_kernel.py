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
        normalize_features: bool = False,
        # ---- Bandwidth ----
        graphbandwidth_init: float = 1.0,
        graphbandwidth_prior: Optional[Prior] = None,
        graphbandwidth_constraint: Optional[Interval] = None,
        learn_graphbandwidth: bool = False,
        # ---- Diffusion scale ----
        diffusion_scale_init: float = 1.0,
        diffusion_scale_prior: Optional[Prior] = None,
        diffusion_scale_constraint: Optional[Interval] = None,
        learn_diffusion_scale: bool = False,
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

        # When True, L2-normalize the feature rows before forming the kernel, i.e.
        # use the cosine / correlation kernel k~(x,y) = <Phi(x),Phi(y)> /
        # (||Phi(x)|| ||Phi(y)||) = cos<(Phi(x),Phi(y)). This forces a constant
        # unit prior variance (diagonal = 1) by quotienting out the per-point
        # feature norm ||Phi(x)||^2 = k(x,x) — which under symmetric Laplacian
        # normalization carries a sqrt(degree)~sqrt(density) sampling artifact.
        # The covariance geometry becomes the ANGULAR spectral distance. Rank is
        # preserved (row-scaling a low-rank matrix stays <= num_modes), so the
        # LowRank/Root operator structure is unchanged. Off by default so the raw
        # kernel behaviour is untouched; the magnitude is then carried by the
        # outputscale (ScaleKernel) instead of the diagonal.
        self.normalize_features = normalize_features

        # When True, gradient-carrying inputs (i.e. LEARNED inducing-point
        # parameters) are routed through the differentiable feature map
        # (`features_differentiable`) instead of the exact eigenvector lookup,
        # whose node-index gather has zero gradient w.r.t. coordinates. Off by
        # default — training/prediction inputs don't require grad and keep the
        # exact path. Flipped on by the model when learn_inducing_locations=True.
        self.differentiable_inducing = False
 
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

        # ---- Freeze the bandwidth by default. The eigval/eigvec passed in are
        #      the spectrum of the Laplacian at `graphbandwidth_init` and are
        #      never recomputed here, so letting the optimizer move the
        #      bandwidth would desync it from the spectrum it defines. Opt in
        #      with learn_graphbandwidth=True only if you recompute eigenpairs.
        if not learn_graphbandwidth:
            self.raw_graphbandwidth.requires_grad_(False)

        # ---- Diffusion scale: a learnable multiplicative factor on the (frozen)
        #      Laplacian spectrum. Reshapes the Matern spectral density via
        #      lambda_k -> diffusion_scale * lambda_k WITHOUT recomputing eigenpairs:
        #      scaling an operator leaves its eigenvectors untouched and only scales
        #      its eigenvalues, so the frozen `eigvec` stay exactly correct. It is the
        #      free, multiplicative companion to the lengthscale's *additive* 2*nu/l^2
        #      floor in S_k = (2*nu/l^2 + diffusion_scale * lambda_k)^-nu. Unlike the
        #      graphbandwidth it does NOT desync the spectrum from the eigenvectors, so
        #      it is safe to learn with frozen eigenpairs. It enters ONLY the spectral
        #      density (the Matern smoothing operator); the OOS Nystrom correction
        #      (1 - graphbandwidth^2 * lambda)^2 stays tied to the raw spectrum, since
        #      that term encodes the graph heat-operator, not the Matern operator.
        #      Init 1.0 (identity); frozen by default so behaviour is unchanged unless
        #      learn_diffusion_scale=True.
        if diffusion_scale_constraint is None:
            diffusion_scale_constraint = Positive()

        self.register_parameter(
            name="raw_diffusion_scale",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
        )

        if diffusion_scale_prior is not None:
            if not isinstance(diffusion_scale_prior, Prior):
                raise TypeError(
                    "Expected gpytorch.priors.Prior but got "
                    + type(diffusion_scale_prior).__name__
                )
            self.register_prior(
                "diffusion_scale_prior", diffusion_scale_prior,
                self._diffusion_scale_param, self._diffusion_scale_closure,
            )

        self.register_constraint("raw_diffusion_scale", diffusion_scale_constraint)
        self._set_diffusion_scale(torch.tensor(float(diffusion_scale_init)))

        if not learn_diffusion_scale:
            self.raw_diffusion_scale.requires_grad_(False)
 
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
    # Diffusion-scale plumbing
    # ----------------------------------------------------------------------
    def _diffusion_scale_param(self, m) -> Tensor:
        return m.diffusion_scale

    def _diffusion_scale_closure(self, m, v: Tensor) -> Tensor:
        return m._set_diffusion_scale(v)

    def _set_diffusion_scale(self, value: Tensor):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_diffusion_scale)
        self.initialize(
            raw_diffusion_scale=self.raw_diffusion_scale_constraint.inverse_transform(value)
        )

    @property
    def diffusion_scale(self) -> Tensor:
        return self.raw_diffusion_scale_constraint.transform(self.raw_diffusion_scale)

    @diffusion_scale.setter
    def diffusion_scale(self, value: Tensor):
        self._set_diffusion_scale(value)


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

        # Cosine / correlation kernel: unit-normalize the feature rows so the
        # kernel is <Phi/||Phi||, Phi/||Phi||>. Done here (not in features()) so
        # it covers every return path below and both the exact and differentiable
        # feature maps uniformly. eps guards the zero-norm rows (points outside
        # all bump support, Phi(x)=0): F.normalize divides by max(||z||, eps), so
        # they stay ~0 (self-covariance 0, i.e. "no support => no information")
        # rather than NaN.
        if self.normalize_features:
            z1 = normalize(z1, dim=-1, eps=1e-12)
            z2 = z1 if x1_eq_x2 else normalize(z2, dim=-1, eps=1e-12)

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
 
    def _interpolated_eigenvectors(self, x: Tensor):
        """UNSCALED interpolated eigenfunctions U(x) ``(B, num_modes)``.

        On-graph rows are the exact eigenvector rows; out-of-sample rows within
        bump support use the Nyström extension (× bump gate); rows beyond support
        stay zero. This is the spectral-density-free core shared by ``features``
        (which re-applies the per-row scale) and ``raw_eigenvectors`` (which
        returns U directly). Returns ``(U, within_global)`` where
        ``within_global`` indexes the OOS rows that received a Nyström value
        (``None`` if there are none) so ``features`` can apply the OOS scale.
        """
        laplacian_ = self.laplacian()

        # faiss GPU search requires a contiguous input; minibatches gathered by a
        # shuffled DataLoader (the spectral / eigenmap paths) can be non-contiguous.
        # No-op when already contiguous, so the manifold path is unaffected.
        if not x.is_contiguous():
            x = x.contiguous()

        # Check if x is a *subset* of knn.x by nearest-neighbor lookup,
        # not torch.equal (which requires identical tensors).
        # For graph nodes: nearest neighbor distance should be exactly 0.
        edge_value_nn, edge_index_nn = self.knn.search(x, 1)
        is_on_graph = (edge_value_nn[:, 0] < 1e-8)   # dist² < 1e-8 → on the graph

        U = torch.zeros(x.shape[0], self.num_modes, device=x.device)
        within_global = None

        # ---- In-sample rows: look up exact eigenvector by node index ----
        if is_on_graph.any():
            node_idx = edge_index_nn[is_on_graph, 0]   # which graph node
            U[is_on_graph] = self.eigvec[node_idx]

        # ---- Out-of-sample rows: Nyström extension ----
        oos = ~is_on_graph
        if oos.any():
            x_oos = x[oos]
            # Full k-NN for Nyström (not just k=1)
            ev_k, ei_k = self.knn.search(x_oos, self.nearest_neighbors)
            within = ev_k[:, 0].sqrt() < self.bump_scale * self.graphbandwidth.squeeze()
            if within.any():
                oos_idx = oos.nonzero(as_tuple=True)[0]
                within_global = oos_idx[within]
                U[within_global] = (
                    laplacian_.out_of_sample(
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

        return U, within_global

    def raw_eigenvectors(self, x: Tensor) -> Tensor:
        """Interpolated UNSCALED Laplacian eigenfunctions φ_k(x) ``(B, num_modes)``.

        The spectral-density-free basis used by the weight-space SpectralLatentGP
        and the eigenmap embedding. ``features(x)`` is this multiplied by the
        per-mode spectral-density scale.
        """
        U, _ = self._interpolated_eigenvectors(x)
        return U

    def features(self, x: Tensor) -> Tensor:
        # Learned-inducing mode: route gradient-carrying inputs (the inducing-
        # point parameters) through the differentiable feature map. Data/test
        # inputs do not require grad, so they keep the exact lookup below. The
        # two maps are mutually consistent (the Nyström correction is designed
        # to extend the on-graph eigenvectors), exactly as features() already
        # mixes exact + Nyström rows within a single call.
        if self.differentiable_inducing and x.requires_grad:
            return self.features_differentiable(x)

        U, within_global = self._interpolated_eigenvectors(x)

        spectral_density = self.spectral_density()
        spectral_density = (spectral_density / spectral_density.sum().clamp(min=1e-12))
        scale = (spectral_density * self.eigvec.shape[0]).sqrt()  # (num_modes,)

        # On-graph rows (and beyond-support zero rows) take the in-sample scale.
        features = U * scale

        # Out-of-sample rows within bump support use the OOS spectral density.
        if within_global is not None:
            sd_oos = self.spectral_density().div(
                (1 - self.graphbandwidth.square() * self.eigval).square().clamp(min=1e-6)
            )
            sd_oos = (sd_oos / sd_oos.sum().clamp(min=1e-12)) * self.eigvec.shape[0]
            scale_oos = sd_oos.sqrt()
            features[within_global] = U[within_global] * scale_oos

        return features

    def features_differentiable(self, x: Tensor) -> Tensor:
        """Feature map that is DIFFERENTIABLE w.r.t. the input coordinates.

        ``features()`` gathers exact eigenvectors by node index for on-graph
        points (``self.eigvec[node_idx]``) — a discrete lookup with ZERO
        gradient w.r.t. ``x`` — and even its Nyström branch reads the neighbour
        distances straight from FAISS (also detached from ``x``). So
        d features / d x ≡ 0 and learned inducing points never move.

        This variant ALWAYS uses the Nyström interpolation and RECOMPUTES the
        neighbour squared-distances in torch from ``x`` and the (constant) graph
        node coordinates, so the heat weights — and hence the interpolated
        eigenvectors — carry a gradient back to ``x``. FAISS is used only to
        pick the neighbour SET (treated as a stop-gradient: the set is
        piecewise-constant in ``x``, so the within-cell gradient is exact almost
        everywhere). Used for LEARNED inducing points; the exact ``features()``
        path is untouched for training/prediction.
        """
        laplacian_ = self.laplacian()

        # Neighbour SELECTION via FAISS — indices only, no gradient through the
        # (locally constant) selection. Detach + contiguous for the FAISS call.
        with torch.no_grad():
            _, ei_k = self.knn.search(x.detach().contiguous(), self.nearest_neighbors)
        ei_k = ei_k.long()

        # Differentiable squared distances: recompute from x and the fixed graph
        # node coordinates the FAISS indices point at. This is the one line that
        # was detached in features(); everything downstream now flows to x.
        nbr = self.knn.x[ei_k]                             # (n, k, d) — constant
        ev_k = (x.unsqueeze(1) - nbr).pow(2).sum(dim=-1)   # (n, k) — grad → x

        # Out-of-sample spectral density (same Nyström correction as the OOS
        # branch of features(), so K_uu is consistent with K_uf / K_ff).
        sd = self.spectral_density().div(
            (1 - self.graphbandwidth.square() * self.eigval).square().clamp(min=1e-6)
        )
        sd = (sd / sd.sum().clamp(min=1e-12)) * self.eigvec.shape[0]
        scale = sd.sqrt()                                 # (num_modes,)

        # Interpolate neighbour eigenvectors with the heat weights, then apply
        # the smooth bump so points beyond bump support decay to zero features
        # (and the gradient only pulls inducing points ALONG the manifold).
        feats = scale * laplacian_.out_of_sample(self.eigvec, ev_k, ei_k)
        # Bump gate on the distance to the nearest graph node. Clamp under the
        # sqrt: an inducing point sitting exactly on a node has nearest distance
        # 0, where d(sqrt)/d· is infinite — and the bump is flat there, so the
        # raw product is a 0·inf NaN in the backward. The floor makes it 0·finite
        # (the gate's gradient at the centre is ~0 anyway, which is correct).
        bump = bump_function(
            ev_k[:, 0].clamp(min=1e-12).sqrt(),
            self.bump_scale * self.graphbandwidth.squeeze(),
            self.bump_decay,
        ).unsqueeze(-1)
        return feats * bump