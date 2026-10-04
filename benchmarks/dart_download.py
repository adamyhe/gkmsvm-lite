"""Auto-download DART-Eval data from Synapse.

Shared helper for DART-Eval benchmark scripts. Downloads only the
specific files needed (data.h5 and input_data/) from each task folder,
not the entire Synapse tree.

Requires: pip install synapseclient
Auth: synapse login (or ~/.synapseConfig)
"""

from __future__ import annotations

import os
import sys

SYNAPSE_IDS = {
    "task_4": ("syn60581041", "task_4_chromatin_activity"),
    "task_5": ("syn60581045", "task_5_variant_effect_prediction"),
    "refs": ("syn60581044", "refs"),
}

NEEDED_FILES = {"data.h5"}
NEEDED_DIRS = {"input_data"}


def _download_selective(syn, folder_id: str, dest: str):
    """Download only data.h5 and input_data/ from a Synapse folder."""
    os.makedirs(dest, exist_ok=True)

    for child in syn.getChildren(folder_id):
        name = child["name"]
        child_id = child["id"]

        if name in NEEDED_FILES:
            target = os.path.join(dest, name)
            if not os.path.exists(target):
                print(f"  Downloading {name} ({child_id})...")
                entity = syn.get(child_id, downloadLocation=dest)
                dl_path = entity.path
                if os.path.abspath(dl_path) != os.path.abspath(target):
                    os.rename(dl_path, target)

        elif name in NEEDED_DIRS:
            subdir = os.path.join(dest, name)
            os.makedirs(subdir, exist_ok=True)
            print(f"  Downloading {name}/ ({child_id})...")
            for subchild in syn.getChildren(child_id):
                sub_target = os.path.join(subdir, subchild["name"])
                if not os.path.exists(sub_target):
                    entity = syn.get(subchild["id"],
                                     downloadLocation=subdir)
                    dl_path = entity.path
                    if os.path.abspath(dl_path) != os.path.abspath(
                            sub_target):
                        os.rename(dl_path, sub_target)


def download_dart_data(work_dir: str,
                       tasks: tuple[str, ...] = ("task_4",)):
    """Download DART-Eval data from Synapse if H5 files are missing."""
    needed = []
    for task in tasks:
        syn_id, subdir = SYNAPSE_IDS[task]
        h5 = os.path.join(work_dir, subdir, "data.h5")
        if not os.path.exists(h5):
            needed.append((syn_id, subdir))

    if not needed:
        return

    try:
        import synapseclient
    except ImportError:
        print(
            "Error: synapseclient required for auto-download.\n"
            "  pip install synapseclient\n"
            "  synapse login\n"
            "Or download manually and set DART_WORK_DIR.",
            file=sys.stderr,
        )
        sys.exit(1)

    syn = synapseclient.Synapse()
    try:
        syn.login(silent=True)
    except Exception:
        print(
            "Error: Synapse authentication failed.\n"
            "  Run: synapse login",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(work_dir, exist_ok=True)
    for syn_id, subdir in needed:
        dest = os.path.join(work_dir, subdir)
        print(f"Downloading {subdir} from Synapse ({syn_id})...")
        _download_selective(syn, syn_id, dest)
