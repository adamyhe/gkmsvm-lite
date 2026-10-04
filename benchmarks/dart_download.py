"""Auto-download DART-Eval data from Synapse.

Shared helper for DART-Eval benchmark scripts. Downloads Task 4
(chromatin activity) and Task 5 (variant effect prediction) H5 files
from Synapse using synapseclient.

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
        import synapseutils
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
        synapseutils.syncFromSynapse(syn, syn_id, path=dest)
