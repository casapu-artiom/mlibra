"""Reference-only parcellation geometry as structure for a GP.

Self-contained: depends on numpy / scipy / scikit-learn / pandas only, and shares
no code with ``manifold_gp`` (no graph Laplacian, no eigensolve, no Riemann
kernel). The coordinate conventions in :mod:`parcelgp.volume` are deliberately
identical to the ones the MALDI parquet is read with, so the two line up.

Pipeline::

    reference_image.npy
        -> features.template_features   multi-scale appearance + shape + depth
        -> parcellate.parcellate        symmetric, contiguous parcels + soft memberships
        -> border.border_distance       distance from each node to its parcel border
        -> field.ParcelField            cached, queryable at any coordinate

    parcelgp.build      build + cache a ParcelField
    parcelgp.validate   score it against the lipid data (the go/no-go gate)
"""
from .border import BorderField, border_distance
from .features import FeatureSpec, template_features
from .field import ParcelField, ParcelSample
from .parcellate import Parcellation, parcellate

__all__ = [
    "FeatureSpec", "template_features",
    "Parcellation", "parcellate",
    "BorderField", "border_distance",
    "ParcelField", "ParcelSample",
]
