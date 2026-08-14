"""One-time: Julia Flux .bson checkpoints -> flat binary + index CSV.

Run from the repo root:
    pip install pymongo numpy
    python3 data-raw/export_weights.py

Only needed when the NBEs are retrained. Users never run this.
"""
import bson, csv, glob, os
import numpy as np

NAMES = ["psi1_W", "psi1_b", "psi2_W", "psi2_b", "phi1_W", "phi1_b",
         "outA_W", "outA_b", "outB_W", "outB_b"]


def collect(node, out):
    """Depth-first walk of a Julia `model_state`, gathering Float32 arrays
    in Flux's serialisation order."""
    if isinstance(node, dict):
        if node.get("tag") == "array":
            out.append((tuple(node["size"]),
                        np.frombuffer(node["data"], dtype=np.float32)))
            return
        if node.get("tag") == "backref":
            return
        for k, v in node.items():
            if k not in ("tag", "type"):
                collect(v, out)
    elif isinstance(node, list):
        for v in node:
            collect(v, out)


rows, blob, off = [], [], 0
for f in sorted(glob.glob("data-raw/nbe/*.bson")):
    key = os.path.basename(f)[:-5]
    arrs = []
    collect(bson.decode(open(f, "rb").read())["model_state"].get("data"), arrs)
    if len(arrs) != 10:
        raise SystemExit(f"{key}: expected 10 arrays, got {len(arrs)}")
    for nm, (size, flat) in zip(NAMES, arrs):
        rows.append([key, nm, size[0], size[1] if len(size) > 1 else 0,
                     off, int(flat.size)])
        blob.append(flat.astype(np.float64))
        off += flat.size

np.concatenate(blob).tofile("data-raw/weights.bin")
with open("data-raw/weights_index.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["model", "param", "d1", "d2", "off", "n"])
    w.writerows(rows)

print(f"exported {len(rows)//10} models, {off} doubles")
