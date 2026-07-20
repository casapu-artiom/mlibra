#!/usr/bin/env python
"""Recompute MISSING per-lipid volumes by re-running INFERENCE for new lipids.

``render_missing.py`` renders volumes that already exist on disk; it has no model
and cannot produce a lipid that was never reconstructed. Because runs pass
``--reconstruction-lipids`` (typically a 5-lipid subset), ``volume*/`` only ever
holds that subset — asking for a sixth lipid needs a forward pass.

This script drives that forward pass. It does NOT reimplement reconstruction:
every trained run already knows how to reconstruct itself without retraining, and
this just re-invokes the run's own entrypoint with a different lipid list:

  * ``experiment_baselines.py`` has ``--skip-training``: loads model.pth + the
    saved stats and jumps straight to reconstruction.
  * the joint LGP entrypoints short-circuit in ``MaldiExperiment.run()`` — it
    loads model.pth when it exists instead of training (experiment.py) — and
    ``train_mean``/``train_std`` are lazy properties that read lipid_means.pth /
    lipid_stds.pth off disk, so no training data is touched.

Every run persists its full parsed-args dict to ``<model>/args.npy`` (config.py
writes it unconditionally), which is what makes re-invocation possible: we
reconstruct the original command line from it, override the lipid list, and hand
it back to the same entrypoint. The rebuilt CLI is round-tripped through that
entrypoint's own ``parse_args()`` and diffed against the intended args before
anything runs, so a mis-rebuilt flag surfaces as an error rather than a subtly
different experiment.

Reconstruction already caches: ``predictions{filter_tag}.npy`` is keyed by voxel
set AND lipid filter, so a new lipid list means a fresh forward pass (expected),
while re-running the same list reuses the cache.

Usage::

    python recompute_missing.py <batch_dir> --lipids "PC 34:1" "PE 38:4" --dry-run
    python recompute_missing.py <batch_dir> --lipids-file my_lipids.txt
    python recompute_missing.py <batch_dir>/one_model --lipids "PC 34:1"
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

from render_missing import _VOLUME_RE

log = logging.getLogger("recompute_missing")

HERE = Path(__file__).resolve().parent

# Entrypoint selection from the saved args dict. Each run's args.npy carries the
# knobs its own parser defined, so the presence of a key identifies the script.
# Ordered most-specific first.
ENTRYPOINTS = [
    # (module basename, predicate, extra flags needed to skip training)
    ("spectral_lgp_manifold_experiment",
     lambda a: "knn_method" in a and _spectralish(a), []),
    ("lgp_manifold_experiment",
     lambda a: "knn_method" in a, []),
    ("experiment_baselines",
     lambda a: "model" in a and "xgb_lr" in a, ["--skip-training"]),
    ("lgp_experiment",
     lambda a: True, []),
]


def _spectralish(a: dict) -> bool:
    return any(k.startswith("spectral") for k in a)


def infer_entrypoint(args: dict) -> tuple[str, list[str]]:
    """(module_name, extra_flags) for the script that produced this args dict."""
    saved = args.get("_entrypoint")
    if saved:
        for name, _pred, extra in ENTRYPOINTS:
            if name == saved:
                return name, extra
    for name, pred, extra in ENTRYPOINTS:
        if pred(args):
            return name, extra
    raise RuntimeError("could not identify the entrypoint for this run")


def existing_lipids(model_dir: Path) -> set[str]:
    """Lipid names that already have a per-lipid volume, across every volume*
    dir (dense ``volume/``, ``volume_sparse/``, ``volume_diffusion/``, ...).
    Uses render_missing's regex so the two agree on what counts as a volume."""
    found: set[str] = set()
    for d in sorted(model_dir.glob("volume*")):
        if not d.is_dir():
            continue
        for f in d.glob("*_volume*.npy"):
            m = _VOLUME_RE.match(f.name)
            if m:
                found.add(m["name"])
    return found


def _flag(dest: str) -> str:
    return "--" + dest.replace("_", "-")


# Keys in args.npy that are NOT CLI flags, or that we set explicitly.
_SKIP_KEYS = {"_entrypoint", "load_args", "skip_training",
              "reconstruction_lipids", "reconstruction_lipids_by_index"}


def args_to_cli(args: dict, lipids: list[str], extra: list[str]) -> list[str]:
    """Rebuild a command line from a saved parsed-args dict.

    Types drive the encoding: bool -> store_true (emitted only when True),
    list/tuple -> nargs, None -> omitted (argparse default). The result is
    validated by round-tripping through the entrypoint's own parser, so an
    encoding mistake fails loudly instead of silently changing the run.
    """
    cli: list[str] = []
    for k, v in sorted(args.items()):
        if k in _SKIP_KEYS or v is None:
            continue
        if isinstance(v, bool):
            if v:
                cli.append(_flag(k))
        elif isinstance(v, (list, tuple)):
            if len(v):
                cli.append(_flag(k))
                cli.extend(str(x) for x in v)
        else:
            cli.extend([_flag(k), str(v)])
    # The whole point: a different lipid list. Names (not indices) so the
    # entrypoint resolves them against its own selected_lipids_names.
    cli.append("--reconstruction-lipids")
    cli.extend(lipids)
    cli.extend(extra)
    return cli


def validate_cli(module: str, cli: list[str], want: dict, lipids: list[str]) -> list[str]:
    """Re-parse the rebuilt CLI with the entrypoint's own parser and diff it
    against the args we meant to reproduce. Returns a list of problems."""
    import importlib
    problems: list[str] = []
    try:
        mod = importlib.import_module(module)
    except Exception as e:  # noqa: BLE001
        return [f"cannot import {module}: {e}"]
    if not hasattr(mod, "parse_args"):
        return [f"{module} has no parse_args() to validate against"]
    old = sys.argv
    try:
        sys.argv = [module] + cli
        got = mod.parse_args()
    except SystemExit as e:
        return [f"{module}.parse_args() rejected the rebuilt CLI (exit {e.code})"]
    except Exception as e:  # noqa: BLE001
        return [f"{module}.parse_args() raised: {e}"]
    finally:
        sys.argv = old
    if not isinstance(got, dict):
        got = vars(got)
    if list(got.get("reconstruction_lipids") or []) != list(lipids):
        problems.append(f"reconstruction_lipids did not round-trip: "
                        f"{got.get('reconstruction_lipids')!r} != {lipids!r}")
    for k, v in want.items():
        if k in _SKIP_KEYS or k not in got:
            continue
        gv = got[k]
        if isinstance(v, (list, tuple)):
            if list(gv or []) != list(v):
                problems.append(f"{k}: {gv!r} != {v!r}")
        elif str(gv) != str(v):
            problems.append(f"{k}: {gv!r} != {v!r}")
    return problems


def find_runs(root: Path) -> list[Path]:
    """Model dirs (including root itself) that carry a saved args.npy."""
    out = []
    for p in [root] + [d for d in root.rglob("*") if d.is_dir()]:
        if (p / "args.npy").exists():
            out.append(p)
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Recompute missing per-lipid volumes by re-running inference.")
    ap.add_argument("batch_dir", type=Path)
    ap.add_argument("--lipids", nargs="+", default=None,
                    help="Lipid names to ensure exist (as in the training script).")
    ap.add_argument("--lipids-file", type=Path, default=None,
                    help="File with one lipid name per line ('#' comments ignored).")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if a volume already exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would run; touch nothing.")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    lipids = list(a.lipids or [])
    if a.lipids_file:
        lipids += [l.strip() for l in a.lipids_file.read_text().splitlines()
                   if l.strip() and not l.startswith("#")]
    if not lipids:
        log.error("Nothing to do: pass --lipids and/or --lipids-file.")
        return 2
    # Preserve order, drop dupes.
    lipids = list(dict.fromkeys(lipids))

    runs = find_runs(a.batch_dir)
    if not runs:
        log.error(f"No runs with args.npy under {a.batch_dir}")
        return 2
    log.info(f"Found {len(runs)} run(s) under {a.batch_dir}")

    n_ok = n_skip = n_fail = 0
    for run in runs:
        try:
            args = np.load(run / "args.npy", allow_pickle=True).item()
        except Exception as e:  # noqa: BLE001
            log.error(f"  {run.name}: unreadable args.npy ({e}); skipping")
            n_fail += 1
            continue

        have = existing_lipids(run)
        want = lipids if a.force else [l for l in lipids if l not in have]
        if not want:
            log.info(f"  {run.name}: all {len(lipids)} lipid(s) present; skipping")
            n_skip += 1
            continue

        try:
            module, extra = infer_entrypoint(args)
        except RuntimeError as e:
            log.error(f"  {run.name}: {e}; skipping")
            n_fail += 1
            continue

        cli = args_to_cli(args, want, extra)
        problems = validate_cli(module, cli, args, want)
        if problems:
            log.error(f"  {run.name}: rebuilt CLI does not round-trip; skipping:")
            for p in problems[:8]:
                log.error(f"      {p}")
            n_fail += 1
            continue

        script = HERE / f"{module}.py"
        cmd = [a.python, str(script)] + cli
        log.info(f"  {run.name}: {module} -> {len(want)} lipid(s): {want}")
        if a.dry_run:
            log.info(f"    [dry-run] {' '.join(cmd[:2])} ... --reconstruction-lipids "
                     f"{' '.join(repr(w) for w in want)} {' '.join(extra)}")
            n_ok += 1
            continue
        r = subprocess.run(cmd, cwd=HERE)
        if r.returncode == 0:
            n_ok += 1
        else:
            log.error(f"  {run.name}: exited {r.returncode}")
            n_fail += 1

    log.info(f"DONE — {n_ok} run(s) {'planned' if a.dry_run else 'recomputed'}, "
             f"{n_skip} already complete, {n_fail} failed.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
