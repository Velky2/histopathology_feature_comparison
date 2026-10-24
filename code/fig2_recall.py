# Fig. 2 (recall por classe no 7k, 100%): barras LBP, LBP-COL, SimCLR, Phikon. Mesmo conteudo da Fig. 2 do paper.
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT56, ROOT
R = pd.read_csv(OUT56 / "semana7" / "recall_por_classe_7k_v2.csv", index_col=0)
B = [("LBP", "LBP", "#9a9a9a"), ("LBP-COR-H", "LBP-COL", "#d1495b"), ("SimCLR_s0", "SimCLR", "#2e6db4"), ("Phikon", "Phikon", "#1b3a5c")]
plt.rcParams.update({"font.size": 7, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(3.45, 1.85)); x = np.arange(len(R)); w = 0.2
for k, (a, n, c) in enumerate(B):
    ax.bar(x + (k - 1.5) * w, R[a].values, w, label=n, color=c)
ax.set_xticks(x); ax.set_xticklabels(R.index, fontsize=6); ax.set_ylabel("recall (%)"); ax.set_ylim(0, 105)
ax.legend(ncol=4, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False); ax.grid(axis="y", alpha=.3)
out = ROOT / "camera_ready" / "figs"; out.mkdir(exist_ok=True, parents=True)
fig.tight_layout(); fig.savefig(out / "fig2_recall_por_classe.pdf"); fig.savefig(out / "fig2_recall_por_classe.png", dpi=200)
print("ok")
