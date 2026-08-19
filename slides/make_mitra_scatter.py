"""Mitra DE-vs-TE scatter for the talk.

Single panel, no grey band. One point is one (layer, task) pair; the dashed
diagonal is TE = DE, i.e. no downstream reaction.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path.home() / "code/tfmlens/src"))
from tfm_lens.evaluation.de_results import de_scale, load_de_json  # noqa: E402

COORD = "margin"
IN = Path.home() / "code/tfmlens/out"


def points(results):
    de, te, layer = [], [], []
    for r in results.values():
        eff = r["effects"]
        scale = de_scale(eff, COORD, agg=True)
        for m, d in eff["de"].items():
            de.append(d[COORD] / scale)
            te.append(eff["te"][m][COORD] / scale)
            layer.append(int(m))
    layer = np.array(layer, float)
    return np.array(de), np.array(te), layer / max(layer.max(), 1)


results = load_de_json(IN, ["mitra"])["mitra"]
de, te, layer = points(results)
lim = float(np.nanmax(np.abs(np.concatenate([de, te])))) * 1.1 + 1e-6

fig, ax = plt.subplots(figsize=(5.8, 5.6), constrained_layout=True)
ax.plot([-lim, lim], [-lim, lim], "--", color="0.4", lw=1.2, zorder=1)
sc = ax.scatter(de, te, c=layer, cmap="viridis", vmin=0, vmax=1, s=34, alpha=0.8, zorder=2)
ax.axhline(0, color="0.75", lw=0.7, zorder=1)
ax.axvline(0, color="0.75", lw=0.7, zorder=1)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.set_xlabel("DE  (direct effect)", fontsize=13)
ax.set_ylabel("TE  (total effect)", fontsize=13)
ax.spines[["top", "right"]].set_visible(False)

out = Path(__file__).with_name("public") / "mitra-de-te.png"
fig.savefig(out, dpi=190, facecolor="white")
print(f"wrote {out}")
