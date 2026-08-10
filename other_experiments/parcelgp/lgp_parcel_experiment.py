"""LGP (latent GPs + MLP decoder) with the parcel factor, on the EXISTING infra.

``maldi/lgp_experiment.py`` is a thin wrapper: it builds a config, picks inducing
points, constructs ``IndependentMultitaskGPModel`` + ``LGP``, and hands them to
``MaldiExperiment``, which owns everything that actually matters -- parquet
loading, negative clipping, the log transform, train-column normalization (saved
and reused for test), checkpoint resume, W&B, train/test prediction, whole-brain
reconstruction and rendering.

So this module does NOT reimplement any of that. It calls their
``setup_experiment`` verbatim and then inserts the parcel factor into the kernel
of the model it returns. Total new logic: about fifteen lines. Everything
downstream is byte-for-byte the existing pipeline, which is the only way the
resulting numbers are comparable to the runs you already have.

    python -m other_experiments.parcelgp.lgp_parcel_experiment \
        --parcel-field .../full_k128_sw1.0.npz \
        <every flag maldi/lgp_experiment.py accepts>

Omit ``--parcel-field`` and this is exactly ``lgp_experiment.py`` -- which is what
makes it a clean ablation baseline rather than an approximate one.

Two things worth knowing about what gets inherited:

  * ``MaldiExperiment.train_fit`` optimizes ``lgp_model.parameters()`` with
    ``AdamW(weight_decay=1e-3)``. ``B`` is a submodule parameter, so it is picked
    up automatically -- and the weight decay acts as a shrink-toward-no-op prior
    on it, which is the right direction for a factor that should stay inert
    unless the data asks otherwise.
  * ``LGP.loss_function`` adds a minibatch-summed reconstruction term to a
    full-dataset KL, so the effective KL weight is ``N/batch`` times the textbook
    ELBO (~1000x at N=5M, batch=4096). That is the convention every existing run
    used, so it is preserved here deliberately rather than silently "fixed" --
    changing it would make these numbers incomparable with every result you
    already have. Worth revisiting as its own experiment.
"""
from __future__ import annotations

import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

# maldi/ must be importable: experiment.py and config.py import each other by
# bare module name, and lgp_experiment.py lives beside them.
_MALDI = Path(__file__).resolve().parents[2] / "maldi"
if str(_MALDI) not in sys.path:
    sys.path.insert(0, str(_MALDI))

from lgp_experiment import parse_args as _base_parse_args           # noqa: E402
from lgp_experiment import setup_experiment as _base_setup          # noqa: E402

from .field import ParcelField                                      # noqa: E402
from .kernels import wrap_scale_kernel                              # noqa: E402


def parse_args() -> dict:
    """The base runner's arguments plus the parcel flags.

    The parcel flags are stripped from ``sys.argv`` first and the remainder is
    handed to the base parser untouched, so any flag added to
    ``lgp_experiment.py`` later works here with no change. (``-h`` therefore
    shows the base runner's help; the parcel flags are documented above.)
    """
    p = ArgumentParser(add_help=False)
    p.add_argument("--parcel-field", dest="parcel_field", default=None,
                   help="parcelgp .npz. Omit for the ablation baseline.")
    p.add_argument("--parcel-rank", dest="parcel_rank", type=int, default=8)
    p.add_argument("--parcel-init-scale", dest="parcel_init_scale", type=float,
                   default=0.05)
    p.add_argument("--parcel-shared-B", dest="parcel_per_task",
                   action="store_false", default=True)
    p.add_argument("--parcel-no-tag", dest="parcel_tag", action="store_false",
                   default=True,
                   help="Do not append the parcel settings to --exp-name.")
    parcel, rest = p.parse_known_args()

    argv = sys.argv
    sys.argv = [argv[0]] + rest
    try:
        args = _base_parse_args()
    finally:
        sys.argv = argv
    args.update(vars(parcel))
    return args


def setup_experiment(args: dict):
    """Build the standard experiment, then wrap its kernel with the parcel factor."""
    field = None
    if args.get("parcel_field"):
        field = ParcelField.load(args["parcel_field"])
        if args.get("parcel_tag", True):
            # EVERY parcel setting that changes the model goes in the directory
            # name. A sweep over rank / init-scale / shared-B must not land two
            # different models in one directory -- MaldiExperiment.run() would
            # then load the first arm's model.pth and skip training entirely, and
            # a shape-compatible mismatch (e.g. same rank, different init scale)
            # would do so SILENTLY. The parcel field's own stem carries its build
            # parameters, which parcelgp.build keeps honest.
            tag = f"parcel{Path(args['parcel_field']).stem}-r{args['parcel_rank']}"
            init = float(args.get("parcel_init_scale", 0.05))
            if abs(init - 0.05) > 1e-12:
                tag += f"-is{init:g}"
            if not args.get("parcel_per_task", True):
                tag += "-sharedB"
            args["exp_name"] = f"{args['exp_name']}-{tag}"

    experiment = _base_setup(args)

    if field is not None:
        gp = experiment.lgp_model.gp_model
        covar = getattr(gp, "covar_module", None)
        if covar is None:
            raise ValueError(
                "the GP has no covar_module to wrap — the parcel factor needs the "
                "inducing-point model built by lgp_experiment.setup_experiment")
        wrap_scale_kernel(
            covar, field, num_tasks=experiment.config.latent_dim,
            rank=int(args["parcel_rank"]),
            init_scale=float(args["parcel_init_scale"]),
            per_task=bool(args.get("parcel_per_task", True)),
        )
        experiment.lgp_model.to(experiment.config.device)
        # Echo the field's RECORDED build parameters, not the filename's claims.
        # normalize_blocks / stride / spatial_weight are properties of the field,
        # not runtime flags, so the run log is the only place they are visible
        # after the fact -- and a hand-renamed field would otherwise leave no
        # trace of what it actually is.
        m = field.meta
        logging.info(
            "parcel factor active: %d parcels, rank %d, %d latent GPs, "
            "init_scale %g, per_task %s",
            field.n_parcels, args["parcel_rank"], experiment.config.latent_dim,
            float(args["parcel_init_scale"]), bool(args.get("parcel_per_task", True)))
        logging.info(
            "  field %s — features=%s stride=%s threshold=%s spatial_weight=%s "
            "normalize_blocks=%s symmetric=%s seed=%s feature_version=%s",
            Path(args["parcel_field"]).name,
            m.get("features", "?"), m.get("stride", "?"), m.get("threshold", "?"),
            m.get("spatial_weight", "?"), m.get("normalize_blocks", "?"),
            m.get("symmetric", "?"), m.get("seed", "?"),
            m.get("feature_version", "?"))
    else:
        logging.info("no --parcel-field: running the plain LGP baseline")

    return experiment


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    experiment = setup_experiment(args)
    experiment.run()
    if experiment.config.do_brain_reconstruction:
        if experiment.config.reconstruction_lipids_by_index:
            lipid_names, lipid_indices = None, experiment.config.reconstruction_lipids
        else:
            lipid_names, lipid_indices = experiment.config.reconstruction_lipids, None
        experiment.whole_brain_reconstruction(lipid_indices=lipid_indices,
                                              lipid_names=lipid_names)


if __name__ == "__main__":
    main()
