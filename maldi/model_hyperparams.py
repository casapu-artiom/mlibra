#!/usr/bin/env python
"""
Dump the TRAINED hyperparameters of a saved GP model.

Point it at a checkpoint (or a run dir / a tree of them) and it prints every
learned kernel/likelihood hyperparameter as its ACTUAL constrained value -- not
the raw pre-constraint tensor GPyTorch stores on disk.

Two checkpoint layouts are understood:
  * whole-brain runs      -> ``<run>/model.pth``            (a bare state_dict, or
                             ``{"state_dict": ...}``)
  * per-lipid runs        -> ``<run>/checkpoints/batch_NNN.pt`` wrapped as
                             ``{"model_state", "likelihood_state", lipid_names,
                             n_tasks, args}``. Each batch holds SEVERAL lipids as
                             per-task kernels (``base_kernel.kernels.<i>``); the
                             table view unrolls them to ONE ROW PER LIPID.

Why the values are exact without rebuilding the model
-----------------------------------------------------
GPyTorch stores each hyperparameter as ``raw_<name>`` and registers its constraint
as a submodule ``raw_<name>_constraint`` whose ``lower_bound`` / ``upper_bound``
are BUFFERS -- so they are in the checkpoint too. The trained value is
``constraint.transform(raw)``:

    * upper == +inf   -> softplus(raw) + lower              (Positive / GreaterThan)
    * upper is finite -> lower + (upper-lower)*sigmoid(raw) (Interval)

If a raw param has no saved bounds we fall back to softplus and flag it with '~'.

Usage
-----
    python maldi/model_hyperparams.py /workspace/output/<EXP_NAME>
    python maldi/model_hyperparams.py /workspace/output --glob         # detail, every run
    python maldi/model_hyperparams.py /workspace/output --table        # compact sweep table
    python maldi/model_hyperparams.py /workspace/output/per_lipid --table   # one row per lipid
    python maldi/model_hyperparams.py <run> --raw                      # also show raw tensors
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


_CONSTRAINT_SUFFIXES = ("_constraint.lower_bound", "_constraint.upper_bound")
_KERNEL_IDX_RE = re.compile(r"kernels\.(\d+)\.")


# ---------------------------------------------------------------------------
# Discovery + loading
# ---------------------------------------------------------------------------
def discover(path: Path) -> list[Path]:
    """All checkpoint files at/under PATH: model.pth and checkpoints/batch_*.pt
    (skipping *_inprogress.pt)."""
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    files = list(path.rglob("model.pth"))
    files += [p for p in path.rglob("checkpoints/batch_*.pt")
              if "inprogress" not in p.name]
    return sorted(set(files))


def run_label(path: Path) -> str:
    """The experiment dir name -- for batch_*.pt that's the grandparent (they live
    under <run>/checkpoints/)."""
    return path.parent.parent.name if path.parent.name == "checkpoints" else path.parent.name


def load_merged(path: Path) -> tuple[dict, dict]:
    """Return (flat_state_dict, meta). Handles a bare state_dict, a
    {"state_dict": ...} wrapper, or the per-lipid {"model_state","likelihood_state",
    ...meta} wrapper (likelihood keys are re-prefixed 'likelihood.' so names and
    constraints stay unambiguous)."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError(f"not a checkpoint dict (got {type(obj).__name__})")
    if "model_state" in obj and isinstance(obj["model_state"], dict):
        sd = dict(obj["model_state"])
        lik = obj.get("likelihood_state")
        if isinstance(lik, dict):
            for k, v in lik.items():
                sd[f"likelihood.{k}"] = v
        meta = {k: v for k, v in obj.items()
                if k not in ("model_state", "likelihood_state")}
        return sd, meta
    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return dict(obj["state_dict"]), {}
    return dict(obj), {}


# ---------------------------------------------------------------------------
# Constraint transforms
# ---------------------------------------------------------------------------
def collect_bounds(sd: dict) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """base_param_key -> (lower_bound, upper_bound) from the *_constraint buffers."""
    lowers, uppers = {}, {}
    for k, v in sd.items():
        if k.endswith("_constraint.lower_bound"):
            lowers[k[: -len("_constraint.lower_bound")]] = torch.as_tensor(v)
        elif k.endswith("_constraint.upper_bound"):
            uppers[k[: -len("_constraint.upper_bound")]] = torch.as_tensor(v)
    return {base: (lowers[base], uppers.get(base)) for base in lowers}


def _bound_str(t: torch.Tensor) -> str:
    """A scalar bound formats as itself; a per-task vector bound as 'lo..hi'."""
    t = t.reshape(-1)
    if t.numel() == 1:
        return f"{float(t[0]):g}"
    return f"{float(t.min()):g}..{float(t.max()):g}"


def transform(raw: torch.Tensor, bounds) -> tuple[torch.Tensor, str]:
    """Apply the constraint transform; return (value, constraint_label). Bounds may
    be scalars OR per-task vectors (they broadcast against raw)."""
    raw = raw.float()
    if bounds is None:
        return F.softplus(raw), "~positive(assumed)"
    lower, upper = bounds
    lower = lower.float()
    if upper is None or bool(torch.isinf(upper).all()):
        label = "positive" if bool((lower.reshape(-1) == 0).all()) \
            else f"greater_than({_bound_str(lower)})"
        return F.softplus(raw) + lower, label
    upper = upper.float()
    return lower + (upper - lower) * torch.sigmoid(raw), \
        f"interval({_bound_str(lower)},{_bound_str(upper)})"


def fmt_tensor(t: torch.Tensor, max_elems: int) -> str:
    shape = tuple(t.shape)
    t = t.detach().float().flatten()
    n = t.numel()
    if n == 0:
        return "<empty>"
    if n == 1:
        return f"{t.item():.6g}"
    if n <= max_elems:
        return "[" + ", ".join(f"{x:.4g}" for x in t.tolist()) + "]"
    return (f"shape={shape} n={n}  min={t.min():.4g} mean={t.mean():.4g} "
            f"max={t.max():.4g} std={t.std():.4g}")


def clean_name(raw_key: str) -> str:
    """covar_module.base_kernel.raw_lengthscale -> base_kernel.lengthscale."""
    return raw_key.replace("raw_", "").replace("covar_module.", "")


def task_col(raw_key: str) -> str:
    """...base_kernel.kernels.3.raw_lengthscale -> lengthscale (drop the per-task prefix)."""
    return re.sub(r".*kernels\.\d+\.", "", raw_key).replace("raw_", "")


# ---------------------------------------------------------------------------
# --table row extraction (one row per lipid when the model is per-task)
# ---------------------------------------------------------------------------
def _is_param(k: str, v) -> bool:
    return torch.is_tensor(v) and "raw_" in k and not any(
        k.endswith(sfx) for sfx in _CONSTRAINT_SUFFIXES)


def _put_vector(row: dict, name: str, val: torch.Tensor, expand_max: int) -> None:
    """Scalar -> one column; a SHORT vector (<= expand_max) -> one column per
    element (name.0, name.1, ...); a longer vector -> a single name[mean] column."""
    val = val.flatten()
    if val.numel() == 1:
        row[name] = round(float(val.item()), 6)
    elif val.numel() <= expand_max:
        for j, x in enumerate(val.tolist()):
            row[f"{name}.{j}"] = round(float(x), 6)
    elif val.numel() <= 4096:
        row[f"{name}[mean]"] = round(float(val.mean()), 6)


def extract_rows(path: Path, expand_max: int = 8) -> list[dict]:
    base = {"run": run_label(path)}
    try:
        sd, meta = load_merged(path)
    except Exception as ex:  # noqa: BLE001
        return [{**base, "error": str(ex)}]
    bounds = collect_bounds(sd)

    task_idx = sorted({int(m.group(1)) for k in sd for m in [_KERNEL_IDX_RE.search(k)] if m})
    lipid_names = meta.get("lipid_names")
    n_tasks = meta.get("n_tasks") or (len(task_idx) if task_idx else None)

    # Per-lipid unrolling ONLY when the checkpoint declares lipid_names (the
    # per-lipid pipeline). A whole-brain lgp/manifold model also has kernels.<i>,
    # but there the <i> are LATENT DIMENSIONS, not lipids -- so it stays one flat
    # row with the per-dim kernels kept as distinct columns.
    per_lipid = bool(lipid_names) and (bool(task_idx) or (n_tasks or 0) > 1)

    if not per_lipid:
        row = dict(base)
        for k, v in sd.items():
            if not _is_param(k, v):
                continue
            # A short vector hyperparameter (e.g. per-latent-GP outputscale) is
            # expanded to one column per element rather than averaged.
            _put_vector(row, clean_name(k), transform(v, bounds.get(k))[0], expand_max)
        return [row]

    # --- per-lipid: one row per lipid (kernels.<i> may be absent if the batch
    # shares one base kernel -- then only the per-task vectors differ) ---
    task_range = task_idx if task_idx else list(range(n_tasks))
    rows = []
    for i in task_range:
        row = dict(base)
        row["batch"] = path.stem
        row["lipid"] = str(lipid_names[i]) if i < len(lipid_names) else f"task{i}"
        for k, v in sd.items():
            if not _is_param(k, v):
                continue
            m = _KERNEL_IDX_RE.search(k)
            if m:                                     # per-task kernel param
                if int(m.group(1)) != i:
                    continue
                val = transform(v, bounds.get(k))[0].flatten()
                if val.numel() == 1:
                    row[task_col(k)] = round(float(val.item()), 6)
            else:                                     # shared scalar, or per-task vector
                val = transform(v, bounds.get(k))[0].flatten()
                if val.numel() == 1:
                    row[clean_name(k)] = round(float(val.item()), 6)
                elif n_tasks and val.numel() == n_tasks:
                    row[clean_name(k)] = round(float(val[i].item()), 6)
        rows.append(row)
    return rows


def report_table(models: list[Path], csv_path: Path | None = None,
                 expand_max: int = 8) -> None:
    import pandas as pd
    rows = [r for m in models for r in extract_rows(m, expand_max=expand_max)]
    df = pd.DataFrame(rows)
    front = [c for c in ("run", "batch", "lipid", "task") if c in df.columns]
    tail = ["error"] if "error" in df.columns else []
    mid = [c for c in df.columns if c not in front + tail]
    df = df[front + mid + tail]
    if front:
        df = df.sort_values(front, kind="stable")
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Wrote {len(df)} row(s) x {len(df.columns)} col(s) to {csv_path}")
        return
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 260)
    pd.set_option("display.float_format", lambda v: f"{v:.5g}")
    print(f"### TRAINED HYPERPARAMETERS  ({len(models)} checkpoint(s), {len(df)} row(s))")
    print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# --glob / single detailed view
# ---------------------------------------------------------------------------
def report_model(model_path: Path, args) -> None:
    sd, meta = load_merged(model_path)
    bounds = collect_bounds(sd)

    hyper_rows, other_rows = [], []
    for k, v in sd.items():
        if any(k.endswith(sfx) for sfx in _CONSTRAINT_SUFFIXES):
            continue
        if not torch.is_tensor(v):
            other_rows.append((k, str(v)))
            continue
        if "raw_" in k:
            val, label = transform(v, bounds.get(k))
            hyper_rows.append((clean_name(k), fmt_tensor(val, args.max_elems), label))
            if args.raw:
                hyper_rows.append(("  (raw) " + clean_name(k),
                                   fmt_tensor(v, args.max_elems), ""))
        else:
            other_rows.append((k, fmt_tensor(v, args.max_elems)))

    print("=" * 100)
    print(f"MODEL: {model_path}")
    if meta.get("lipid_names") is not None:
        print(f"lipids ({meta.get('n_tasks', len(meta['lipid_names']))}): "
              f"{', '.join(map(str, meta['lipid_names']))}")
    # Config context: prefer the checkpoint's own args, else the sibling args.npy.
    cfg = meta.get("args")
    if not isinstance(cfg, dict):
        args_npy = model_path.parent / "args.npy"
        if args_npy.exists():
            try:
                a = np.load(args_npy, allow_pickle=True)
                cfg = a.item() if getattr(a, "shape", None) == () else a
            except Exception:  # noqa: BLE001
                cfg = None
    if isinstance(cfg, dict):
        keys = ["exp_name", "kernel_family", "kernel", "nu", "stride", "num_modes",
                "knn_method", "laplacian_norm", "learn_diffusion_scale",
                "diffusion_scale_init", "graphbandwidth_init", "num_inducing", "latent_dim"]
        ctx = "  ".join(f"{k}={cfg[k]}" for k in keys if k in cfg)
        if ctx:
            print(f"config: {ctx}")
    print("=" * 100)

    if not hyper_rows:
        print("  (no raw_* hyperparameters found -- is this a GPyTorch checkpoint?)")
    else:
        w = max(len(r[0]) for r in hyper_rows)
        print("### TRAINED HYPERPARAMETERS (constrained values)")
        for name, val, label in hyper_rows:
            tail = f"   [{label}]" if label else ""
            print(f"  {name:<{w}}  {val}{tail}")

    if other_rows:
        print("\n### OTHER LEARNED TENSORS (means / inducing points / variational params)")
        w = max(len(r[0]) for r in other_rows)
        for name, val in other_rows:
            print(f"  {name:<{w}}  {val}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path,
                    help="A checkpoint file, a run dir, or (with --glob/--table) a root.")
    ap.add_argument("--glob", action="store_true",
                    help="Detailed report for EVERY checkpoint beneath PATH.")
    ap.add_argument("--table", action="store_true",
                    help="Compact table of scalar learned hyperparameters across all "
                         "checkpoints beneath PATH (one row per lipid for per-lipid runs).")
    ap.add_argument("--csv", type=Path, default=None, metavar="PATH",
                    help="Write the table to this CSV instead of printing it (implies --table).")
    ap.add_argument("--expand-max", type=int, default=8, metavar="N",
                    help="In the table, vectors with <= N elements (e.g. per-latent-GP "
                         "outputscale) get one column each (name.0, name.1, ...); longer "
                         "ones collapse to name[mean]. Set 1 to always average (default 8).")
    ap.add_argument("--raw", action="store_true",
                    help="Also print the raw (pre-constraint) tensor for each hyperparameter.")
    ap.add_argument("--max-elems", type=int, default=16,
                    help="Vectors longer than this are summarised instead of printed (default 16).")
    args = ap.parse_args()

    models = discover(args.path)
    if not models:
        print(f"No checkpoints (model.pth / checkpoints/batch_*.pt) found under {args.path}",
              file=sys.stderr)
        return 1

    if args.table or args.csv:
        report_table(models, csv_path=args.csv, expand_max=args.expand_max)
        return 0

    # Bare single path spanning >1 run: nudge toward the sweep flags, but proceed.
    if not args.glob and len({run_label(m) for m in models}) > 1:
        print(f"  note: {len(models)} checkpoints across "
              f"{len({run_label(m) for m in models})} runs; use --table for a compact "
              f"sweep view or --glob for full detail.\n", file=sys.stderr)

    for m in models:
        try:
            report_model(m, args)
        except Exception as ex:  # noqa: BLE001
            print(f"  ! failed on {m}: {ex}", file=sys.stderr)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
