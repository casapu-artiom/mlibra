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

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, **kwargs) -> Tensor:
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        x1_eq_x2 = torch.equal(x1, x2)
        z1 = self.features(x1)
        if not x1_eq_x2:
            z2 = self.features(x2)
        else:
            z2 = z1

        if diag:
            return (z1 * z2).sum(-1)
        if x1_eq_x2:
            # Exploit low rank structure, if there are fewer features than data points
            if z1.size(-1) < z2.size(-2):
                return LowRankRootLinearOperator(z1)
            else:
                return RootLinearOperator(z1)
        else:
            return MatmulLinearOperator(z1, z2.transpose(-1, -2))

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
        if torch.equal(x, self.knn.x):
            spectral_density = self.spectral_density()
            spectral_density /= spectral_density.sum()
            return (spectral_density * self.eigvec.shape[0]).sqrt() * self.eigvec
 
        edge_value, edge_index = self.knn.search(x, self.nearest_neighbors)
        x_within_support = edge_value[:, 0].sqrt() < self.bump_scale * self.graphbandwidth.squeeze()
        features = torch.zeros(x.shape[0], self.num_modes, device=x.device)
 
        if x_within_support.sum() != 0:
            spectral_density = self.spectral_density().div(
                (1 - self.graphbandwidth.square() * self.eigval).square()
            )
            spectral_density /= spectral_density.sum()
            spectral_density *= self.knn.x.shape[0]
 
            features[x_within_support] = (
                spectral_density.sqrt()
                * laplacian_.out_of_sample(
                    self.eigvec, edge_value[x_within_support],
                    edge_index[x_within_support],
                )
                * bump_function(
                    edge_value[x_within_support, 0].sqrt(),
                    self.bump_scale * self.graphbandwidth.squeeze(),
                    self.bump_decay,
                ).unsqueeze(-1)
            )
 
        return features
 
    def raw_eigenvectors(self, x: Tensor) -> Tensor:
        """Pure eigenvectors evaluated at x, without spectral density scaling.
 
        Used as the geometric basis functions for Spectral Latent GPs.
        """
        laplacian_ = self.laplacian()
 
        if torch.equal(x, self.knn.x):
            return self.eigvec
 
        edge_value, edge_index = self.knn.search(x, self.nearest_neighbors)
        x_within_support = edge_value[:, 0].sqrt() < self.bump_scale * self.graphbandwidth.squeeze()
        eigvecs_out = torch.zeros(x.shape[0], self.num_modes, device=x.device)
 
        if x_within_support.sum() != 0:
            projected = laplacian_.out_of_sample(
                self.eigvec,
                edge_value[x_within_support],
                edge_index[x_within_support],
            )
            decay = bump_function(
                edge_value[x_within_support, 0].sqrt(),
                self.bump_scale * self.graphbandwidth.squeeze(),
                self.bump_decay,
            ).unsqueeze(-1)
            eigvecs_out[x_within_support] = projected * decay
 
        return eigvecs_out
 

    # def eval(self):
    #     if getattr(self, 'cached_eigval', None) is not None and getattr(self, 'cached_eigvec', None) is not None:
    #         self.eigvec = self.cached_eigvec
    #         self.eigval = self.cached_eigval
    #         return super().eval()
    #     # self.laplacian_operator = self.laplacian()
    #     # with torch.no_grad():
    #     #     import scipy.sparse as sp
    #     #     import scipy.sparse.linalg as sla

    #     #     # 1. Extract sparse data to CPU
    #     #     N = self.laplacian_operator.operator_dimension
    #     #     idx = self.laplacian_operator.idx.cpu()
    #     #     val = self.laplacian_operator.laplacian_triu.cpu()
    #     #     diag = self.laplacian_operator.laplacian_diag.cpu()

    #     #     # 2. Build SciPy CSR Matrix (stays sparse, no OOM)
    #     #     row, col = idx[0].numpy(), idx[1].numpy()
    #     #     edge_data = -val.numpy() 
    #     #     L_scipy = sp.csr_matrix((edge_data, (row, col)), shape=(N, N))
    #     #     L_scipy = L_scipy + L_scipy.T 
    #     #     L_scipy.setdiag(diag.numpy())

    #     #     # 3. Fix 2: Shift-and-Invert (Targets smooth anatomical modes)
    #     #     # sigma=0 finds eigenvalues closest to zero efficiently
    #     #     evals_np, evecs_np = sla.eigsh(
    #     #         L_scipy, k=self.num_modes, sigma=0, which='LM'
    #     #     )

    #     #     # 4. Move final results back to GPU
    #     #     self.eigval = torch.from_numpy(evals_np).float().cuda()
    #     #     self.eigvec = torch.from_numpy(evecs_np).float().cuda()

    #     #     # 5. Mandatory Post-processing from the paper
    #     #     self.eigval[0] = 0.0  # Theoretically zero for Laplacian
    #     #     self.eigvec *= self.laplacian_operator.degree_mat.pow(-0.5).view(-1, 1)
    #     #     self.eigvec = normalize(self.eigvec, p=2, dim=0)

    #     self.laplacian_operator = self.laplacian()
    #     with torch.no_grad():
    #         #self.eigval, self.eigvec = self.laplacian_operator.diagonalization(num_modes=self.num_modes)
    #         # idx = torch.cat((torch.arange(self.laplacian_operator.operator_dimension, device=self.laplacian_operator.device).repeat(2, 1),
    #         #                  self.laplacian_operator.idx, torch.stack((self.laplacian_operator.idx[1], self.laplacian_operator.idx[0]), dim=0)), dim=1)
    #         # val = torch.cat((self.laplacian_operator.laplacian_diag, -self.laplacian_operator.laplacian_triu.repeat(2)))
    #         #self.eigval, self.eigvec = torch.linalg.eigh(torch.sparse_coo_tensor(idx, val, [self.laplacian_operator.operator_dimension, self.laplacian_operator.operator_dimension]).to_dense())
    #         # L_sparse = torch.sparse_coo_tensor(
    #         #     idx, val, 
    #         #     [self.laplacian_operator.operator_dimension, self.laplacian_operator.operator_dimension]
    #         # ).cuda()

    #         # L_sparse = L_sparse.coalesce().to_sparse_csr()

    #         # pbar = tqdm(desc="    -> Torch LOBPCG", leave=False)

    #         # def progress_tracker(lobpcg_state):
    #         #     pbar.update(1)

    #         # evals, evecs = torch.lobpcg(
    #         #     L_sparse, 
    #         #     k=self.num_modes, 
    #         #     largest=False, 
    #         #     tol=1e-4,
    #         #     tracker=progress_tracker  # Inject the progress bar here
    #         # )
    #         # pbar.close()
    #         # self.eigval, self.eigvec = evals, evecs

    #         print("AAAA")
    #         N = self.laplacian_operator.operator_dimension
    #         idx = self.laplacian_operator.idx.cpu()
    #         val = self.laplacian_operator.laplacian_triu.cpu()
    #         diag = self.laplacian_operator.laplacian_diag.cpu()

    #         # # 2. Build SciPy CSR Matrix (Handles the sparse structure perfectly)
    #         row, col = idx[0].numpy(), idx[1].numpy()
    #         edge_data = -val.numpy() 
    #         L_scipy = sp.csr_matrix((edge_data, (row, col)), shape=(N, N))
    #         L_scipy = L_scipy + L_scipy.T 
    #         L_scipy.setdiag(diag.numpy())

    #         # --- NEW SPARSITY PRINT ---
    #         nnz = L_scipy.nnz
    #         total_elements = N * N
    #         density = (nnz / total_elements) * 100
    #         sparsity = 100.0 - density
    #         print(f"\n[Graph Stats] Nodes: {N:,} | Non-Zero Entries (Edges + Diag): {nnz:,}")
    #         print(f"[Graph Stats] Density: {density:.6f}% | Sparsity: {sparsity:.6f}%\n")
    #         # --------------------------

    #         print("BBBb")
    #         # # 3. Transfer to CuPy (Moves the sparse structure to GPU memory)
    #         L_cp = cpsparse.csr_matrix(L_scipy)

    #         # # 4. Initialize the SpMV Progress Tracker
    #         class CuPyProgressTracker(cplinalg.LinearOperator):
    #             def __init__(self, A):
    #                 super().__init__(A.dtype, A.shape)
    #                 self.A = A
    #                 self.pbar = tqdm(desc="    -> CuPy Lanczos SpMV", leave=False)

    #             def _matvec(self, x):
    #                 self.pbar.update(1)
    #                 return self.A @ x
                    
    #         op_with_progress = CuPyProgressTracker(L_cp)

    #         # # 5. Solve using CuPy Lanczos
    #         # # We bump ncv to help the solver distinguish clustered eigenvalues
    #         ncv_target = min(N, max(2 * self.num_modes + 1, 1500)) 
            
    #         print("CCCCC")
    #         evals_cp, evecs_cp = cplinalg.eigsh(
    #             op_with_progress, 
    #             k=self.num_modes, 
    #             which='SA',    # Smallest Algebraic (CuPy doesn't support sigma=0)
    #             tol=1e-4,      # Looser tolerance to prevent infinite iteration loops
    #             ncv=ncv_target
    #         )
    #         # [FIX] Clamp negative numerical noise caused by disconnected components
    #         evals_cp = cp.clip(evals_cp, a_min=0.0, a_max=None)

    #         cp.cuda.Stream.null.synchronize()
    #         op_with_progress.pbar.close()

    #         print("DDDDD")

    #         self.eigval, self.eigvec = evals_cp, evecs_cp

    #         if hasattr(self.laplacian_operator, "_memoize_cache"):
    #             self.laplacian_operator._memoize_cache.clear()
        
    #         # # 2. Delete the temporary construction tensors
    #         # # This is critical if these were defined in this scope
    #         if 'L_sparse' in locals(): del L_sparse
    #         if 'idx' in locals(): del idx
    #         if 'val' in locals(): del val
            
    #         # # 3. Force PyTorch to release "Reserved" memory back to the OS
    #         import gc
    #         gc.collect()
    #         torch.cuda.empty_cache()

    #         if not isinstance(self.eigvec, torch.Tensor):
    #             if hasattr(self.eigvec, 'toDlpack'):
    #                 # Zero-copy transfer for CuPy arrays
    #                 self.eigval = torch.from_dlpack(self.eigval.toDlpack()).float()
    #                 self.eigvec = torch.from_dlpack(self.eigvec.toDlpack()).float()
    #             else:
    #                 # Standard transfer for NumPy arrays
    #                 self.eigval = torch.from_numpy(self.eigval).float().cuda()
    #                 self.eigvec = torch.from_numpy(self.eigvec).float().cuda()

    #         self.eigval, self.eigvec = self.eigval[:self.num_modes], self.eigvec[:, :self.num_modes]
    #         self.eigval[0] = 0.0
    #         #self.eigvec *= self.laplacian_operator.degree_mat.pow(-0.5).view(-1, 1)
    #         degree_safe = torch.clamp(self.laplacian_operator.degree_mat, min=1e-8)
    #         self.eigvec *= degree_safe.pow(-0.5).view(-1, 1)

    #         self.eigvec = normalize(self.eigvec, p=2, dim=0)


            
    #     return super().eval()

    # def features(self, x: Tensor) -> Tensor:
    #     laplacian_ = self.laplacian()
    #     if torch.equal(x, self.knn.x):
    #         spectral_density = self.spectral_density()
    #         spectral_density /= spectral_density.sum()
    #         return (spectral_density * self.eigvec.shape[0]).sqrt() * self.eigvec
    #     else:
    #         edge_value, edge_index = self.knn.search(x, self.nearest_neighbors)
    #         x_within_support = edge_value[:, 0].sqrt() < self.bump_scale*self.graphbandwidth.squeeze()
    #         features = torch.zeros(x.shape[0], self.num_modes, device=x.device)

    #         if x_within_support.sum() != 0:
    #             spectral_density = self.spectral_density().div((1 - self.graphbandwidth.square() * self.eigval).square())
    #             spectral_density /= spectral_density.sum()
    #             spectral_density *= self.knn.x.shape[0]

    #             features[x_within_support] = spectral_density.sqrt() * laplacian_.out_of_sample(self.eigvec, edge_value[x_within_support], edge_index[x_within_support]) * \
    #                 bump_function(edge_value[x_within_support, 0].sqrt(), self.bump_scale*self.graphbandwidth.squeeze(), self.bump_decay).unsqueeze(-1)

    #         return features

    # def raw_eigenvectors(self, x: Tensor) -> Tensor:
    #     """
    #     Returns the pure eigenvectors evaluated at x, without spectral density scaling.
    #     Used as the pure geometric basis functions for Spectral Latent GPs.
    #     """
    #     laplacian_ = self.laplacian()
        
    #     # 1. In-Sample: Return the exact calculated eigenvectors
    #     if torch.equal(x, self.knn.x):
    #         return self.eigvec
            
    #     # 2. Out-of-Sample: Nyström Extension
    #     else:
    #         edge_value, edge_index = self.knn.search(x, self.nearest_neighbors)
    #         x_within_support = edge_value[:, 0].sqrt() < self.bump_scale * self.graphbandwidth.squeeze()
            
    #         eigvecs_out = torch.zeros(x.shape[0], self.num_modes, device=x.device)

    #         if x_within_support.sum() != 0:
    #             # Project eigenvectors to the new points
    #             projected_eigvecs = laplacian_.out_of_sample(
    #                 self.eigvec, 
    #                 edge_value[x_within_support], 
    #                 edge_index[x_within_support]
    #             )
                
    #             # Apply the bump function to smoothly decay points near the manifold edge
    #             bump_decay_weights = bump_function(
    #                 edge_value[x_within_support, 0].sqrt(), 
    #                 self.bump_scale * self.graphbandwidth.squeeze(), 
    #                 self.bump_decay
    #             ).unsqueeze(-1)
                
    #             eigvecs_out[x_within_support] = projected_eigvecs * bump_decay_weights

    #         return eigvecs_out