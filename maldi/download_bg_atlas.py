from argparse import ArgumentParser
import logging
import numpy as np
from pathlib import Path

def maybe_download_template_allensdk(self, volume_path, template_file):
    if not template_file.exists():
        logging.info("Downloading template volume...")
        from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
        # Specify the resolution you want for the template volume (in microns)
        resolution_um = 25
        mcc = MouseConnectivityCache(resolution=resolution_um)
        logging.info(f"Downloading/loading reference TEMPLATE volume at {resolution_um} um resolution...")
        template_volume, _ = mcc.get_template_volume()
        logging.info(f"Template volume shape: {template_volume.shape}")
        logging.info(f"Template volume data type: {template_volume.dtype}")

        volume_path.mkdir(parents=True, exist_ok=True)
        np.save(template_file, template_volume)
    else:
        logging.info("Template volume already exists, loading from file")
        template_volume = np.load(template_file)
    return template_volume

def coarsen_annotation(annotation, atlas, max_depth=4):
    """Collapse leaf labels to a chosen ancestor depth in the structure tree."""
    structures = atlas.structures
    id_remap = {0: 0}
    for sid, info in structures.items():
        if sid == 0:
            continue
        path = info.get("structure_id_path", [sid])
        if len(path) <= max_depth + 1:
            id_remap[sid] = sid
        else:
            id_remap[sid] = path[max_depth]
    max_id = int(annotation.max()) + 1
    lut = np.zeros(max_id + 1, dtype=np.int32)
    for src, dst in id_remap.items():
        if src < lut.shape[0]:
            lut[src] = dst
    return lut[annotation]

def _load_or_download_template(reference_file: Path, anotations_file: Path, max_depth: int = 4):
    from bg_atlasapi.bg_atlas import BrainGlobeAtlas
    atlas = BrainGlobeAtlas("allen_mouse_25um")

    if not reference_file.exists() or not anotations_file.exists():
        logging.info("Downloading template volume via BrainGlobe Atlas API...")
        template_volume = atlas.reference
        annotation_volume = coarsen_annotation(atlas.annotation, atlas, max_depth=max_depth)
        logging.info(f"Template volume shape: {template_volume.shape}")
        reference_file.parent.mkdir(parents=True, exist_ok=True)
        anotations_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(reference_file, template_volume)
        np.save(anotations_file, annotation_volume)

if __name__ == "__main__":
    parser = ArgumentParser(description="Download BrainGlobe atlas.")
    parser.add_argument("--reference-file", dest="reference_file", type=Path, required=True, help="Path to the reference file.")
    parser.add_argument("--annotations-file", dest="annotations_file", type=Path, required=True, help="Path to the annotations file.")
    # The Allen tree bottoms out at path length 10, so max_depth >= 9 is the
    # raw leaf annotation (672 labels) and anything larger is identical.
    parser.add_argument("--max-depth", dest="max_depth", type=int, default=4,
                        help="Structure-tree depth to collapse leaves to: "
                             "1 -> 5 labels, 2 -> 17, 3 -> 41, 4 -> 86, 5 -> 176, "
                             "6 -> 347, 7 -> 477, 8 -> 629, 9+ -> 672 (leaves).")
    args = parser.parse_args()

    _load_or_download_template(args.reference_file, args.annotations_file, args.max_depth)
