# ============================================================================
# B1.4 - Varredura de dimensao do PCA (pos-aceite, exploratoria; ver adendo_posaceite.md)
#   d in {8, 16, 64, 128} (so se d <= dimensao original e d <= n_treino) e PCA retendo
#   90% / 95% da variancia; fracoes 5% (5 sementes) e 100%; probe identico ao do paper.
#   Predicoes novas do 7k vao para camera_ready/out/preds_posaceite/ (nunca para preds_7k/).
#
#   python b1_4_pca_sweep.py        [N_PROC=4]
# ============================================================================
import os, sys, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ARMS_SWEEP = ["Fractal", "Fractal-BM", "LBP", "LBP-H", "LBP-HF-H", "LBP-COR-H", "LBP-ENR",
              "CLASSICO-MAX", "SimCLR_abl_s0", "SimCLR_s0", "Phikon"]
CONDS = [8, 16, 64, 128, 0.90, 0.95]
FRACS = [0.05, 1.0]


def cname(c):
    return f"PCA{c}" if isinstance(c, int) else f"EVR{int(round(c * 100))}"


def _one(arm, cond, frac, seed):
    os.environ.setdefault("OMP_NUM_THREADS", "3")
    from common import (arm_features, fit_probe, predict, y_7k, bal_acc, CR_PRED,
                        label_subsample, LABEL_FRACTIONS)
    pp = CR_PRED / f"B14__{arm}__{cname(cond)}__f{int(frac*100)}__s{seed}.npy"
    X100, X7, tf = arm_features(arm)
    if isinstance(cond, int) and (cond > X100.shape[1] or cond > len(label_subsample(frac, seed))):
        return None
    kp = pp.with_name(pp.stem + "_k.txt")
    if pp.exists() and kp.exists():
        yp = np.load(pp).astype(int); k = int(kp.read_text())
    else:
        p = fit_probe(X100, frac, seed, cond, tf)
        yp = predict(p, X7, tf)
        k = int(p.named_steps["pca"].n_components_)
        np.save(pp, yp.astype(np.int8)); kp.write_text(str(k))
    return dict(arm=arm, cond=cname(cond), frac=frac, seed=seed, k=k, dim=int(X100.shape[1]),
                bal_acc=bal_acc(y_7k, yp))


if __name__ == "__main__":
    from common import *
    OUT = CR_OUT / "B1_4"; OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = [(a, c, f, s) for a in ARMS_SWEEP for c in CONDS for f in FRACS for s in seeds_for(f)]
    # 100% primeiro (mais caros), para equilibrar a carga
    jobs.sort(key=lambda j: -j[2])
    rows = Parallel(n_jobs=int(os.environ.get("N_PROC", 4)), verbose=5)(delayed(_one)(*j) for j in jobs)
    R = pd.DataFrame([r for r in rows if r is not None])
    R.to_csv(OUT / "B1_4_bruto.csv", index=False)

    df7 = pd.read_csv(OUT56 / "resultados_7k_FINAL.csv")
    raw = df7[(df7.cond == "RAW") & df7.frac.isin(FRACS)].groupby(["arm", "frac"]).bal_acc.mean()
    p32 = df7[(df7.cond == "PCA32") & df7.frac.isin(FRACS)].groupby(["arm", "frac"]).bal_acc.mean()
    M = R.groupby(["arm", "cond", "frac"]).agg(bal_acc=("bal_acc", "mean"), k=("k", "max")).reset_index()
    extra = []
    for (a, f), v in p32.items():
        if a in ARMS_SWEEP: extra.append(dict(arm=a, cond="PCA32", frac=f, bal_acc=v, k=32))
    for (a, f), v in raw.items():
        if a in ARMS_SWEEP: extra.append(dict(arm=a, cond="RAW", frac=f, bal_acc=v, k=None))
    M = pd.concat([M, pd.DataFrame(extra)], ignore_index=True)
    M["custo"] = M.apply(lambda x: raw[(x.arm, x.frac)] - x.bal_acc, axis=1)
    M.to_csv(OUT / "B1_4_medias.csv", index=False)

    ordem = ["RAW", "PCA8", "PCA16", "PCA32", "PCA64", "PCA128", "EVR90", "EVR95"]
    tab = M.pivot_table(index=["frac", "arm"], columns="cond", values="bal_acc")[[c for c in ordem if c in set(M.cond)]]
    custo = M.pivot_table(index=["frac", "arm"], columns="cond", values="custo")[[c for c in ordem if c in set(M.cond)]]
    kk = M[M.cond.str.startswith("EVR")].pivot_table(index=["frac", "arm"], columns="cond", values="k")

    # gap aprendido - classico em cada celula; paridade restaurada?
    G = []
    for f in FRACS:
        for c in ordem:
            for L in ["SimCLR_s0", "Phikon"]:
                for C in CLASSICOS:
                    try:
                        g = M[(M.arm == L) & (M.cond == c) & (M.frac == f)].bal_acc.iloc[0] - \
                            M[(M.arm == C) & (M.cond == c) & (M.frac == f)].bal_acc.iloc[0]
                    except IndexError:
                        continue
                    G.append(dict(frac=f, cond=c, aprendido=L, classico=C, gap=round(g, 2)))
    G = pd.DataFrame(G); G.to_csv(OUT / "B1_4_gaps.csv", index=False)
    menor = G.loc[G.groupby(["frac", "cond"]).gap.idxmin()]
    txt = ["# B1.4 - varredura de dimensao (7k, pos-aceite)\n",
           "## Acuracia balanceada (media sobre sementes em 5%)\n", tab.round(2).to_markdown(),
           "\n\n## Custo (RAW - condicao), pontos\n", custo.round(2).to_markdown(),
           "\n\n## k escolhido pela PCA de variancia retida\n", kk.to_markdown(),
           "\n\n## Menor gap aprendido - classico por (fracao, condicao)\n", menor.to_markdown(index=False),
           f"\n\nAlguma celula com gap <= 0? {bool((G.gap <= 0).any())}"]
    (OUT / "B1_4_resumo.md").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt)); print(f"fim {time.time()-t0:.0f}s")
