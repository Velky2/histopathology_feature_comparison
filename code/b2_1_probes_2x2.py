# ============================================================================
# B2.1 - Probes das celulas novas do 2x2 do SimCLR (CPU, depois do job de GPU)
#   Mesmo protocolo do toque no 7k (RAW e PCA32 x 5 fracoes x 5 sementes) + probe de 100%
#   no split interno (para a amplificacao interno -> externo) + recall por classe.
#   Le camera_ready/cache_2x2/simclr_<tag>_{100k,7k}.npy (extrai_gpu_camera_ready.py --modo ckpt).
#
#   python b2_1_probes_2x2.py s0_crop_semcor s0_semcrop_comcor [mais tags...]   [N_PROC=8]
# ============================================================================
import os, sys, json, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
C2 = None


def _fit(tag, cond, frac, seed):
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    from common import ROOT, fit_probe, predict, y_7k, bal_acc, CR_PRED, idx_eval_int, y_int, recalls_pt
    d = ROOT / "camera_ready" / "cache_2x2"
    X100 = np.load(d / f"simclr_{tag}_100k.npy", mmap_mode="r"); X7 = np.load(d / f"simclr_{tag}_7k.npy")
    pca = None if cond == "RAW" else 32
    pp = CR_PRED / f"B21__SimCLR_{tag}__{cond}__f{int(frac*100)}__s{seed}.npy"
    out = dict(tag=tag, cond=cond, frac=frac, seed=seed)
    if pp.exists() and not (cond == "RAW" and frac == 1.0):
        yp = np.load(pp).astype(int)
    else:
        p = fit_probe(X100, frac, seed, pca, None)
        yp = predict(p, X7); np.save(pp, yp.astype(np.int8))
        if cond == "RAW" and frac == 1.0:          # interno (amplificacao) so no probe de 100%
            yi = predict(p, np.asarray(X100[idx_eval_int]))
            r_i, _ = recalls_pt(y_int, yi)
            out.update({f"int_{k}": float(v) for k, v in enumerate(r_i)})
    r7, cm = recalls_pt(y_7k, yp)
    out.update(bal_acc=float(r7.mean()), **{f"ext_{k}": float(v) for k, v in enumerate(r7)},
               BACK_para_ADI=float(100 * cm[1, 0] / cm[1].sum()))
    return out


if __name__ == "__main__":
    from common import *
    tags = sys.argv[1:] or ["s0_crop_semcor", "s0_semcrop_comcor"]
    falta = [t for t in tags if not (ROOT / "camera_ready" / "cache_2x2" / f"simclr_{t}_7k.npy").exists()]
    if falta:
        print("[aviso] sem features (celula falhou ou nao rodou):", falta)
    tags = [t for t in tags if t not in falta]
    assert tags, "nenhuma celula com features em camera_ready/cache_2x2/"
    OUT = CR_OUT / "B2_1"; OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(t, c, f, s) for t in tags for c in ["RAW", "PCA32"] for f in LABEL_FRACTIONS for s in seeds_for(f)]
    jobs.sort(key=lambda j: -j[2])
    R = pd.DataFrame(Parallel(n_jobs=int(os.environ.get("N_PROC", 8)), verbose=5)(delayed(_fit)(*j) for j in jobs))
    R.to_csv(OUT / "B2_1_bruto.csv", index=False)

    # tabela 2x2 (100%, RAW) com as celulas ja existentes (preds publicadas)
    df7 = pd.read_csv(OUT56 / "resultados_7k_FINAL.csv")
    m = df7.groupby(["arm", "cond", "frac"]).bal_acc.mean()
    int_ref = {"SimCLR_s0": 97.96}
    rI = CR_OUT / "interno" / "tabela2_9_e_8_classes.csv"
    if rI.exists():
        T2 = pd.read_csv(rI, index_col=0); int_ref["SimCLR_abl_s0"] = T2.loc["SimCLR_abl_s0", "int9"]
        int_ref["Phikon"] = T2.loc["Phikon", "int9"]; int_ref["LBP-COR-H"] = T2.loc["LBP-COR-H", "int9"]
    rows = []
    for nome, arm, crop, cor in [("completo (paper)", "SimCLR_s0", "sim", "sim"),
                                 ("sem crop e sem cor = SimCLR-abl (paper)", "SimCLR_abl_s0", "nao", "nao")]:
        rows.append(dict(celula=nome, crop=crop, cor=cor,
                         **{f"ext_{int(f*100)}": round(m[(arm, "RAW", f)], 2) for f in LABEL_FRACTIONS},
                         ext_PCA32_100=round(m[(arm, "PCA32", 1.0)], 2), int_100=int_ref.get(arm)))
    for t in tags:
        r = R[R.tag == t]; g = r.groupby(["cond", "frac"]).bal_acc.mean()
        i100 = r[(r.cond == "RAW") & (r.frac == 1.0)].iloc[0]
        rows.append(dict(celula=t, crop="sim" if "crop_" in t and "semcrop" not in t else ("nao" if "semcrop" in t else "sim"),
                         cor="nao" if "semcor" in t else "sim",
                         **{f"ext_{int(f*100)}": round(g[("RAW", f)], 2) for f in LABEL_FRACTIONS},
                         ext_PCA32_100=round(g[("PCA32", 1.0)], 2),
                         int_100=round(np.mean([i100[f"int_{k}"] for k in range(9)]), 2),
                         rec_BACK_ext=round(i100["ext_1"], 2), BACK_para_ADI=round(i100["BACK_para_ADI"], 2)))
    T = pd.DataFrame(rows)
    T["queda_int_ext"] = (T.int_100 - T.ext_100).round(2)
    if "Phikon" in int_ref:
        phi_i, phi_e = int_ref["Phikon"], m[("Phikon", "RAW", 1.0)]
        T["ampl_vs_Phikon"] = ((phi_e - T.ext_100) - (phi_i - T.int_100)).round(2)
    T["custo_PCA32_100"] = (T.ext_100 - T.ext_PCA32_100).round(2)
    c0 = T.iloc[0]                       # completo (paper)
    T["ampl_vs_completo"] = ((c0.ext_100 - T.ext_100) - (c0.int_100 - T.int_100)).round(2)
    MUD = {"s0_crop_semcor": "sem cor (jitter+gray)", "s0_semcrop_comcor": "sem crop",
           "s0_full_semgray": "sem grayscale", "s0_full_semrot": "sem rot90"}
    T.insert(1, "mudanca", [MUD.get(c, "-") for c in T.celula])
    T.to_csv(OUT / "B2_1_tabela_2x2.csv", index=False)
    conv = []
    for t in tags:
        j = OUT / f"convergencia_{t}.json"
        if j.exists(): conv.append(json.loads(j.read_text()))
    txt = ["# B2.1 - ablacao 2x2 do SimCLR (semente 0)\n", T.to_markdown(index=False),
           "\n\n## Convergencia das celulas novas\n", pd.DataFrame(conv).to_markdown(index=False) if conv else "_sem json_",
           "\n\nCelulas antigas: completo s0 = loss 2,31 / kNN 87,2 na ep. 20; sem crop+sem cor s0 = loss 1,63, "
           "kNN 54,7 (ep. 5, melhor) -> 51,4 (ep. 25)."]
    (OUT / "B2_1_resumo.md").write_text("\n".join(txt), encoding="utf-8"); print("\n".join(txt))
