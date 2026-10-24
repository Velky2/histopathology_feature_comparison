# ============================================================================
# Fig. 1 do paper (interno x externo, 100% dos rotulos) com a correcao E9: o interno do FRAC
# passa a usar a MESMA extracao do externo (FRACTALV1). Os demais internos sao os publicados
# (h3b_escada_classica_interno.csv; Phikon 99.80, SimCLR 97.96). O FRAC interno sai do refit
# (camera_ready/out/interno/tabela2_9_e_8_classes.csv; no HPC, no ambiente original).
#   python fig1_corrigida.py   -> camera_ready/figs/fig1_interno_externo.pdf/.png
# ============================================================================
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT56, CR_OUT, ROOT

h3b = pd.read_csv(OUT56 / "h3b_escada_classica_interno.csv")
interno = h3b[h3b.frac == 1.0].set_index("arm").bal_acc.to_dict()
interno.update({"Phikon": 99.80, "SimCLR_s0": 97.96})
T2 = pd.read_csv(CR_OUT / "interno" / "tabela2_9_e_8_classes.csv", index_col=0)
interno["Fractal"] = float(T2.loc["Fractal", "int9"])          # correcao E9
df7 = pd.read_csv(OUT56 / "resultados_7k_FINAL.csv")
externo = df7[(df7.cond == "RAW") & (df7.frac == 1.0)].set_index("arm").bal_acc.to_dict()

ORD = [("Fractal", "FRAC", 38, "g"), ("Fractal-BM", "FRAC-BM", 35, "g"), ("LBP", "LBP", 54, "g"),
       ("LBP-H", "LBP-H", 54, "g"), ("LBP-HF-H", "LBP-HF", 176, "g"), ("LBP-COR-H", "LBP-COL", 246, "c"),
       ("LBP-ENR", "LBP-ENR", 422, "c"), ("CLASSICO-MAX", "CLASSIC-MAX", 457, "c"),
       ("SimCLR_s0", "SimCLR", 2048, "l"), ("Phikon", "Phikon", 768, "l")]
COR = {"g": ("#3a3a3a", "o"), "c": ("#d1495b", "s"), "l": ("#2e6db4", "^")}

plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(3.45, 2.9))
for k, (arm, nome, d, g) in enumerate(ORD):
    y = len(ORD) - 1 - k
    c, mk = COR[g]
    xi, xe = interno[arm], externo[arm]
    ax.plot([xe, xi], [y, y], color=c, lw=1.2, zorder=1)
    ax.scatter([xi], [y], marker=mk, s=26, facecolors="white", edgecolors=c, linewidths=1.1, zorder=3)
    ax.scatter([xe], [y], marker=mk, s=26, color=c, zorder=3)
    ax.text(xe - 0.6, y + 0.18, f"−{xi - xe:.1f}", color=c, fontsize=6.5, ha="right", va="center")
ax.set_yticks(range(len(ORD)))
ax.set_yticklabels([f"{n} ({d})" for _, n, d, _ in ORD][::-1], fontsize=7)
ax.set_xlim(55, 101); ax.set_xlabel("balanced accuracy (%)")
ax.grid(axis="x", alpha=.3)
ax.scatter([], [], marker="o", facecolors="white", edgecolors="k", label="internal (same cohort)")
ax.scatter([], [], marker="o", color="k", label="external (CRC-VAL-HE-7K)")
ax.legend(fontsize=5.5, loc="lower left", bbox_to_anchor=(0.0, 0.0), frameon=True, borderpad=0.3, handletextpad=0.3)
ax.set_ylim(-1.6, len(ORD) - 0.4)
out = ROOT / "camera_ready" / "figs"; out.mkdir(parents=True, exist_ok=True)
fig.tight_layout(); fig.savefig(out / "fig1_interno_externo.pdf"); fig.savefig(out / "fig1_interno_externo.png", dpi=200)
print("FRAC interno =", interno["Fractal"], "| quedas:",
      {n: round(interno[a] - externo[a], 2) for a, n, _, _ in ORD})
