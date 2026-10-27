import json, sys, shutil, numpy as np, trimesh
d, rfrac, ylo, yhi = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
m = json.load(open(f"{d}/meta.json"))
c = np.array([x["position"] for x in m["cameras"]]); cc = c.mean(0)
R = rfrac * np.linalg.norm((c - cc)[:, [0, 2]], axis=1).mean()
src = f"{d}/scene.full.glb"
import os
if not os.path.exists(src): shutil.copy(f"{d}/scene.glb", src)
sc = trimesh.load(src, force="scene")
out = trimesh.Scene()
for name, g in sc.dump(concatenate=False) and [(n, gg) for n, gg in zip(range(10**6), sc.dump(concatenate=False))]:
    v = g.vertices
    keep_v = (np.hypot(v[:, 0] - cc[0], v[:, 2] - cc[2]) < R) & (v[:, 1] > ylo) & (v[:, 1] < yhi)
    keep_f = keep_v[g.faces].all(1)
    if keep_f.sum() == 0: continue
    g2 = g.submesh([np.nonzero(keep_f)[0]], append=True); g2.remove_unreferenced_vertices()
    out.add_geometry(g2)
out.export(f"{d}/scene.glb")
b = out.bounds
m["bbox_min"], m["bbox_max"] = b[0].round(4).tolist(), b[1].round(4).tolist()
json.dump(m, open(f"{d}/meta.json", "w"), indent=2)
print("R", round(R, 2), "faces", sum(len(g.faces) for g in out.geometry.values()), "bbox", m["bbox_min"], m["bbox_max"])
