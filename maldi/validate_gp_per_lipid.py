import argparse
import logging
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import gpytorch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from config import MaldiConfig
from experiment import MaldiExperiment
from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel

from utils import (
    get_inducing_points,
    get_bbox_inducing_points,
    apply_region_to_config,
    crop_or_stride_volume,
    reference_ccf_from_subvolume,
)

from manifold_gp.operators.graph_laplacian_operator import GraphLaplacianOperator
from manifold_gp.utils.compute_eigenvectors import (
    LaplacianEigensolver, make_key as make_eig_key,
)
from manifold_gp.utils.nearest_neighbors import KnnGraphCache, make_key as make_graph_key

class VariationalGP(gpytorch.models.ApproximateGP):
    """
    Standard Single-Task Variational GP using Inducing Points.
    This mimics the core GP layer of the LGP experiment, allowing 
    minibatch training over the entire dataset.
    """
    def __init__(self, inducing_points, kernel):
        # We use a standard Cholesky distribution for the variational posterior
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        
        # learn_inducing_locations=False is CRITICAL for the Riemann kernel so 
        # inducing points don't drift off the defined anatomical graph.
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=False
        )
        super().__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        # ScaleKernel learns the output scale (variance) of the kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_variational_gp(model, likelihood, dataloader, epochs, device, desc="Training"):
    """Standard Minibatch Training Loop for Variational GPs"""
    model.train()
    likelihood.train()
    
    # Optimize both the GP hyperparameters (lengthscale, etc) and the variational parameters
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.01)
    
    # VariationalELBO handles the KL-divergence scaling automatically based on num_data
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(dataloader.dataset))
    
    pbar = tqdm(range(epochs), desc=desc)
    for epoch in pbar:
        epoch_loss = 0.0
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        pbar.set_postfix({'Loss': f"{epoch_loss / len(dataloader):.4f}"})

def evaluate_variational_gp(model, likelihood, test_x, test_y, batch_size, device):
    """Batched Evaluation to prevent memory spikes on large test sets"""
    model.eval()
    likelihood.eval()
    
    preds_list = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, test_x.size(0), batch_size):
            batch_x = test_x[i : i + batch_size].to(device)
            # Pass through the likelihood to get the predictive distribution
            preds = likelihood(model(batch_x))
            preds_list.append(preds.mean.cpu())
            
    all_preds = torch.cat(preds_list, dim=0)
    test_y_cpu = test_y.cpu()
    
    mse = torch.nn.functional.mse_loss(all_preds, test_y_cpu).item()
    corr = np.corrcoef(all_preds.numpy(), test_y_cpu.numpy())[0, 1]
    
    return mse, corr

def parse_args():
    parser = argparse.ArgumentParser(description="Validate Riemann Kernel vs Euclidean Kernel with Variational GPs.")
    parser.add_argument("--mode", type=str, default="validate", help="Experiment mode.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--maldi-file", type=str, required=True, help="Path to the MALDI file.")
    parser.add_argument("--template-name", dest="template_name", type=str, required=True, help="The reference image npy.")
    parser.add_argument("--reference-file", dest="reference_file", type=str, required=True, help="The reference image npy.")
    parser.add_argument("--annotations-file", dest="annotations_file", type=str, help="The annotations if needed.")
    parser.add_argument("--exp-name", type=str, required=True, help="Name of the experiment.")
    parser.add_argument("--available-lipids-file", type=str, required=True, help="File with available lipids.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for output files.")
    parser.add_argument("--slices-dataset-file", type=str, required=True, help="File for slices dataset.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on.")
    parser.add_argument("--kernel", type=str, default="ignore", help="Not used.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--eigenvector-dir", dest="eigenvector_dir", type=str, required=True, help="Directory for eigenvector files.")

    # Validation-specific arguments
    parser.add_argument("--num-lipids", dest="num_lipids", type=int, default=10, help="Number of lipids to test.")
    parser.add_argument("--num-inducing", type=int, default=500, help="Number of inducing points.")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs to train each Variational GP.")
    parser.add_argument("--manifold-stride", type=int, default=4, help="Stride for subsampling the template volume.")
    parser.add_argument("--batch-size", type=int, default=2000, help="Minibatch size.")
    parser.add_argument("--latent-dim", type=int, default=10, help="Ignored.")
    parser.add_argument("--n-pixels", type=int, default=10, help="Ignored.")
    parser.add_argument("--nu", type=float, default=1.0, help="Smoothness parameter for the Riemann GP model.")
    parser.add_argument("--knn-method", dest="knn_method", type=str, default="faiss",
                        choices=["faiss", "anatomical_atlas"])
    parser.add_argument("--laplacian-norm", dest="laplacian_norm", type=str, default="symmetric",
                        choices=["symmetric", "randomwalk"])
    parser.add_argument("--stride", dest="stride", type=int, default=4, help="Stride to downsample the template.")
    parser.add_argument("--knn-k", dest="knn_k", type=int, default=15, help="Number of knn neighbours for the Graph Laplacian.")
    parser.add_argument("--n-list", type=int, default=1, help="FAISS nlist parameter")
    parser.add_argument("--bump-scale", dest="bump_scale", type=float, default=3.0, help="Bump function param.")
    parser.add_argument("--bump-decay", dest="bump_decay", type=float, default=0.05, help="Bump function param.")
    parser.add_argument("--num-modes", dest="num_modes", type=int, default=200, help="Number of eigenvectors to use.")

    # ---- region restriction ----
    parser.add_argument(
        "--region-bbox", dest="region_bbox", type=int, nargs=6, default=None,
        metavar=("ZMIN", "ZMAX", "YMIN", "YMAX", "XMIN", "XMAX"),
        help=("Optional bbox in voxel coords of the full-res 25um atlas. "
              "If set: the atlas is cropped at full resolution (stride is "
              "ignored), inducing points are placed inside the bbox via "
              "k-means, and MALDI train/test points are filtered to the "
              "same bbox in mm."),
    )

    # Dummy args to satisfy MaldiConfig
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--log-transform", action='store_true')
    parser.add_argument("--use-diffusion", action='store_true')
    
    return vars(parser.parse_args())

class EuclideanMultitaskVGP(gpytorch.models.ApproximateGP):
    """One SVGP per lipid, batched. Each task has its own lengthscale,
    outputscale, mean, and variational posterior. Inducing points shared."""
    def __init__(self, inducing_points, num_tasks, nu=2.5):
        var_dist = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0), batch_shape=torch.Size([num_tasks])
        )
        var_strat = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, var_dist,
                learn_inducing_locations=False,
            ),
            num_tasks=num_tasks,
        )
        super().__init__(var_strat)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_tasks])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=nu, batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks]),
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class BatchedRiemannWrapper(gpytorch.kernels.Kernel):
    """Strip the task-batch dim before calling RiemannMaternKernel, then add it
    back. The Riemann kernel itself is shared across tasks; only the wrapping
    ScaleKernel's outputscale is batched."""
    def __init__(self, base_kernel):
        super().__init__()
        self.base_kernel = base_kernel

    def forward(self, x1, x2, diag=False, **params):
        is_batch = x1.dim() == 3
        if is_batch:
            x1 = x1[0]
        if x2.dim() == 3:
            x2 = x2[0]
        res = self.base_kernel.forward(x1, x2, diag=diag, **params)
        if is_batch and not diag:
            res = res.unsqueeze(0)
        return res


class RiemannMultitaskVGP(gpytorch.models.ApproximateGP):
    """Multitask SVGP where every task uses the SAME Riemann kernel (with the
    same graphbandwidth/Matern hyperparameters), but each task has its own
    outputscale, variational posterior, and mean."""
    def __init__(self, inducing_points, num_tasks, manifold_kernel):
        var_dist = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0), batch_shape=torch.Size([num_tasks])
        )
        var_strat = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, var_dist,
                learn_inducing_locations=False,
            ),
            num_tasks=num_tasks,
        )
        super().__init__(var_strat)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_tasks])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            BatchedRiemannWrapper(manifold_kernel),
            batch_shape=torch.Size([num_tasks]),
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )

def train_multitask_vgp(model, likelihood, train_x, train_y, batch_size,
                         epochs, device, desc, lr=0.01):
    """train_x: (N, 3) on GPU. train_y: (N, num_tasks) on GPU."""
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()), lr=lr
    )
    n = train_x.size(0)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n)

    pbar = tqdm(range(epochs), desc=desc)
    for _ in pbar:
        perm = torch.randperm(n, device=device)
        loss_sum, nb = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            x_b, y_b = train_x[idx], train_y[idx]
            optimizer.zero_grad()
            loss = -mll(model(x_b), y_b)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item(); nb += 1
        pbar.set_postfix({"loss": f"{loss_sum / nb:.4f}"})


def evaluate_multitask_vgp(model, likelihood, test_x, test_y, batch_size, device):
    """Returns per-task MSE and Pearson correlation arrays of shape (num_tasks,)."""
    model.eval(); likelihood.eval()
    preds = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, test_x.size(0), batch_size):
            output = likelihood(model(test_x[i:i + batch_size]))
            preds.append(output.mean.cpu())
    pred = torch.cat(preds, dim=0)               # (n_test, num_tasks)
    truth = test_y.cpu()                          # (n_test, num_tasks)

    mse_per_task = ((pred - truth) ** 2).mean(dim=0).numpy()
    p_c = pred - pred.mean(dim=0, keepdim=True)
    t_c = truth - truth.mean(dim=0, keepdim=True)
    denom = (p_c.pow(2).sum(dim=0).sqrt() *
             t_c.pow(2).sum(dim=0).sqrt() + 1e-10)
    corr_per_task = ((p_c * t_c).sum(dim=0) / denom).numpy()
    return mse_per_task, corr_per_task

def run_comparison(args, config, device, inducing_points, train_x, test_x,
                   train_data, test_data,  knn_k, num_modes,
                   bump_scale, bump_decay, laplacian_norm, graphbandwidth_init,
                   edge_index, edge_value, eigval, eigvec):
    num_lipids = min(args.get("num_lipids"), len(config.selected_lipids_names))
    lipid_names = list(config.selected_lipids_names[:num_lipids])

    # --- pre-load all data to GPU once ---
    train_x = train_x.to(device).contiguous()
    test_x = test_x.to(device).contiguous()
    train_y_all = train_data[:, :num_lipids].to(device).contiguous()
    test_y_all = test_data[:, :num_lipids].to(device).contiguous()
    logging.info(
        f"On-GPU: train_x {tuple(train_x.shape)}, train_y {tuple(train_y_all.shape)}, "
        f"test_x {tuple(test_x.shape)}, test_y {tuple(test_y_all.shape)}"
    )

    batch_size = args.get("batch_size")
    epochs = args.get("epochs")

    # ====================================
    # A. Euclidean multitask VGP
    # ====================================
    logging.info(">>> Training Euclidean multitask VGP "
                 f"(num_tasks={num_lipids})...")
    euc_model = EuclideanMultitaskVGP(
        inducing_points.to(device), num_tasks=num_lipids, nu=2.5
    ).to(device)
    euc_lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_lipids
    ).to(device)
    train_multitask_vgp(
        euc_model, euc_lik, train_x, train_y_all,
        batch_size, epochs, device, desc="Euclidean MT-VGP",
    )
    euc_mse, euc_corr = evaluate_multitask_vgp(
        euc_model, euc_lik, test_x, test_y_all, batch_size, device,
    )

    # ====================================
    # B. Riemann multitask VGP (kernel shared, outputscale batched)
    # ====================================
    logging.info(">>> Training Riemann multitask VGP "
                 f"(num_tasks={num_lipids})...")
    # Single Riemann kernel instance, shared across all tasks via BatchedRiemannWrapper
    from manifold_gp.kernels.riemann_matern_kernel import RiemannMaternKernel
    rie_kernel = RiemannMaternKernel(
        nu=config.nu,
        edge_index=edge_index,
        edge_value=edge_value,
        eigval=eigval,
        eigvec=eigvec,
        nearest_neighbors=knn_k,
        num_modes=num_modes,
        bump_scale=bump_scale,
        bump_decay=bump_decay,
        laplacian_normalization=laplacian_norm,
        graphbandwidth_init=graphbandwidth_init,
    ).to(device)
    rie_kernel.eval()

    rie_model = RiemannMultitaskVGP(
        inducing_points.to(device), num_tasks=num_lipids,
        manifold_kernel=rie_kernel,
    ).to(device)
    rie_lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_lipids
    ).to(device)
    train_multitask_vgp(
        rie_model, rie_lik, train_x, train_y_all,
        batch_size, epochs, device, desc="Riemann MT-VGP",
    )
    rie_mse, rie_corr = evaluate_multitask_vgp(
        rie_model, rie_lik, test_x, test_y_all, batch_size, device,
    )

    # ====================================
    # Results
    # ====================================
    df = pd.DataFrame({
        "Lipid": lipid_names,
        "Euc_MSE": euc_mse,
        "Euc_Corr": euc_corr,
        "Riemann_MSE": rie_mse,
        "Riemann_Corr": rie_corr,
    })
    df["MSE_Diff (Rie - Euc)"] = df["Riemann_MSE"] - df["Euc_MSE"]
    df["Corr_Diff (Rie - Euc)"] = df["Riemann_Corr"] - df["Euc_Corr"]
    return df

def main():
    args = parse_args()
    device = args.get("device")
    torch.manual_seed(args.get("seed"))
    np.random.seed(args.get("seed"))
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 1. Setup Config and load standard normalizations & Inducing Points
    config = MaldiConfig.from_args(args)

    region_bbox = args.get("region_bbox", None)

    # 1. Patch the config's MALDI parquet filters so train/test only sees
    #    points inside the bbox. No-op when region_bbox is None.
    apply_region_to_config(config, region_bbox)


    logging.info("Calculating coordinate normalization factors and inducing points...")
    if region_bbox is not None:
        inducing_points, coord_mean, coord_std = get_bbox_inducing_points(
            config.exp_path, config.dataset_path, config.num_inducing, region_bbox,
        )
        config.num_inducing = inducing_points.shape[0]
    else:
        inducing_points, coord_mean, coord_std = get_inducing_points(
            config.exp_path, config.dataset_path, config.num_inducing,
        )
    logging.info(f"Got {inducing_points.shape[0]} inducing points")
    
    experiment = MaldiExperiment(config, lgp_model=None, coord_mean=coord_mean, coord_std=coord_std)
    
    # 2. Load Train & Test Folds
    logging.info("Loading full training and testing data...")
    experiment.load_train_data()
    experiment.load_coord_train_data()
    experiment.load_test_data()
    experiment.load_coord_test_data()
    
    train_x = experiment.coordinates_train
    test_x = experiment.coordinates_test
    

    # 3. Atlas: download (if needed), coarsen annotations, then crop or stride.
    volume_path = config.reference_file
    template_volume = np.load(config.reference_file)
    annotations_volume = np.load(config.annotations_file)

    threshold            = 5
    stride               = args.get("stride", 4)
    knn_k                = args.get("knn_k", 15)
    nlist                = args.get("n_list", 1)
    num_modes            = args.get("num_modes", 200)
    bump_scale           = args.get("bump_scale", 20.0)
    bump_decay           = args.get("bump_scale", 0.01)
    laplacian_norm       = args.get("laplacian_norm")
    graphbandwidth_init  = 1.0   # used for both eigensolve and kernel init
    knn_method           = args.get("knn_method", "faiss")

    sub_volume, sub_atlas, voxel_offset, voxel_scale_mm = crop_or_stride_volume(
        template_volume, annotations_volume, stride, region_bbox,
    )
    reference_ccf = reference_ccf_from_subvolume(
        sub_volume, voxel_offset, voxel_scale_mm, threshold,
    )
    reference_nodes = torch.tensor(reference_ccf, dtype=torch.float32)
    reference_nodes = (reference_nodes - coord_mean) / coord_std
    reference_nodes = reference_nodes.to(config.device).contiguous()

    # 4. Build (or load) the KNN graph.
    eigenvector_dir = Path(args.get("eigenvector_dir"))
    eigenvector_dir.mkdir(parents=True, exist_ok=True)

    graph_cache_dir = eigenvector_dir / "knn"
    graphs = KnnGraphCache(cache_dir=graph_cache_dir, verbose=True)

    graph_key_parts = {
        "template": config.template_name,
        "stride": stride if region_bbox is None else 1,
        "thresh": threshold,
        "method": knn_method,
        "k": knn_k,
        "nlist": nlist,
        "bbox": tuple(region_bbox) if region_bbox is not None else None,
    }

    graph_key = make_graph_key(graph_key_parts)
    logging.info(f"Graph cache key: {graph_key}")

    if knn_method == "faiss":
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key,
            method="faiss",
            coords=reference_nodes,
            k=knn_k,
            nlist=nlist,
            extra=graph_key_parts,
            force_recompute=bool(args.get("force_recompute_graph", False)),
            device=config.device,
        )
    elif knn_method == "anatomical_atlas":
        knn, edge_index, edge_value = graphs.train_or_load(
            key=graph_key,
            method="anatomical_atlas",
            volume=sub_volume,
            threshold=threshold,
            atlas_volume=sub_atlas,
            connectivity=3,
            coords=reference_nodes,
            k=knn_k,
            nlist=nlist,
            extra=graph_key_parts,
            force_recompute=bool(args.get("force_recompute_graph", False)),
            device=config.device,
        )
    else:
        raise ValueError(f"Unknown knn_method: {knn_method!r}")

    # 5. Eigenpairs: compute (or load from cache) before kernel construction.
    eigvec_cache_dir = eigenvector_dir / "eigvecs"
    laplacian_op = GraphLaplacianOperator(
        edge_value, edge_index, knn.x.shape[0],
        torch.tensor(graphbandwidth_init, device=config.device),
        laplacian_norm,
    )

    eigvec_key_parts = {
        "graph": graph_key,
        "norm": laplacian_norm,
        "bw": graphbandwidth_init,
        "modes": num_modes,
    }
    eigvec_key = make_eig_key(eigvec_key_parts)
    logging.info(f"Eigenvector cache key: {eigvec_key}")

    # Lanczos basis must comfortably exceed num_modes; bump as needed.
    ncv_min = max(1500, 3 * num_modes + 20)
    solver = LaplacianEigensolver(
        num_modes=num_modes, backend="cupy", tol=1e-4, ncv_min=ncv_min, verbose=True,
    )
    eigval, eigvec = solver.compute_or_load(
        laplacian_op,
        cache_dir=eigvec_cache_dir,
        key=eigvec_key,
        graphbandwidth=graphbandwidth_init,
        laplacian_normalization=laplacian_norm,
        extra=eigvec_key_parts,
        force_recompute=bool(args.get("force_recompute_eigvecs", False)),
        device=config.device,
    )
    
    # 4. Iterate over the specified number of lipids
    num_modes = min(1000, reference_nodes.shape[0]//2)

    df = run_comparison(
        args, config, device, inducing_points, train_x, test_x,
        experiment.train_data, experiment.test_data,
        knn_k=knn_k, num_modes=num_modes,
        bump_scale=bump_scale, bump_decay=bump_decay,
        laplacian_norm=laplacian_norm,
        graphbandwidth_init=graphbandwidth_init,
        edge_index=edge_index, edge_value=edge_value,
        eigval=eigval, eigvec=eigvec,
    )

    output_file = config.exp_path / "results.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    pd.set_option("display.float_format", lambda x: "%.4f" % x)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()