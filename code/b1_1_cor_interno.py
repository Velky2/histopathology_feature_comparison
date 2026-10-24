# ============================================================================
# B1.1 - Intervencao de cor no split INTERNO (nao toca o 7k)
#
#   extrai : descritores classicos sobre a subamostra interna (500/classe) sob cada desvio
#            de cor de intervencao_cor.CONDS (le pixels do memmap nct100k_u8.mmap)
#   avalia : probes de 100% ajustados no pool de 80k com as features ORIGINAIS, aplicados
#            as features sob desvio. Inclui os bracos aprendidos se as features existirem
#            (geradas por hpc/extrai_gpu_camera_ready.py --modo cor_interno, ou localmente
#            com --device cpu).
#
#   python b1_1_cor_interno.py extrai  [N_WORKERS=12]
#   python b1_1_cor_interno.py avalia
# ============================================================================
import os, sys, time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from intervencao_cor import CONDS, aplica_cond, subamostra_interna

OUT = CR_OUT / "B1_1"; OUT.mkdir(parents=True, exist_ok=True)
MM_PATH = ROOT / "nct100k_u8.mmap"
DESCS = ["LBP54", "LBPHF", "LBPCOR", "FRACTALBM", "FRACTALV1"]
_MM = None


def _mm():
    global _MM
    if _MM is None:
        _MM = np.memmap(MM_PATH, dtype=np.uint8, mode="r", shape=(100000, 224, 224, 3))
    return _MM


def _worker(args):
    from descritores import compute
    cond, idxs = args
    out = {d: [] for d in DESCS}
    for i in idxs:
        img = aplica_cond(np.asarray(_mm()[i]), cond)
        for d in DESCS:
            out[d].append(compute(d, img))
    return {d: np.asarray(v, dtype=np.float32) for d, v in out.items()}


def extrai(n_workers):
    sub = subamostra_interna()
    np.save(OUT / "subamostra_idx.npy", sub)
    for cond in CONDS:
        if all((OUT / f"{d}__{cond}.npy").exists() for d in DESCS):
            print("[cache]", cond); continue
        t = time.time()
        chunks = [(cond, sub[i:i + 64]) for i in range(0, len(sub), 64)]
        parts = {d: [] for d in DESCS}
        with ProcessPoolExecutor(n_workers) as ex:
            for k, r in enumerate(ex.map(_worker, chunks)):
                for d in DESCS: parts[d].append(r[d])
                if (k + 1) % 10 == 0:
                    print(f"  {cond}: {64*(k+1)}/{len(sub)} ({(time.time()-t)/60:.1f} min)", flush=True)
        for d in DESCS:
            np.save(OUT / f"{d}__{cond}.npy", np.concatenate(parts[d]))
        print(f"[ok] {cond} em {(time.time()-t)/60:.1f} min")


# features da subamostra sob desvio, montadas por braco (mesma composicao de arm_features)
def feats_braco(arm, cond, sub):
    if cond == "orig":
        # bracos aprendidos: se o job de GPU gravou a linha de base RECALCULADA (a000), use-a, para
        # que a queda nao inclua o ruido de recalculo (classicos sao bit-exatos contra o cache)
        p0 = OUT / f"{arm}__a000.npy"
        if p0.exists():
            return np.load(p0), None
        X100, _, tf = arm_features(arm)
        return np.asarray(X100[sub]), tf
    def L(d): return np.load(OUT / f"{d}__{cond}.npy")
    if arm in ("LBP", "LBP-H"):  return L("LBP54"), (hellinger if arm == "LBP-H" else None)
    if arm == "LBP-HF-H":        return L("LBPHF"), hellinger
    if arm == "LBP-COR-H":       return L("LBPCOR"), hellinger
    if arm == "LBP-ENR":         return np.concatenate([hellinger(L("LBPHF")), hellinger(L("LBPCOR"))], 1), None
    if arm == "CLASSICO-MAX":    return np.concatenate([hellinger(L("LBPHF")), hellinger(L("LBPCOR")), L("FRACTALBM")], 1), None
    if arm == "Fractal":         return L("FRACTALV1"), None
    if arm == "Fractal-BM":      return L("FRACTALBM"), None
    p = OUT / f"{arm}__{cond}.npy"                 # bracos aprendidos (GPU)
    return (np.load(p), None) if p.exists() else (None, None)


def avalia():
    sub = np.load(OUT / "subamostra_idx.npy")
    ysub = y_100k[sub]
    arms = CLASSICOS + [a for a in ["SimCLR_s0", "SimCLR_s1", "SimCLR_s2", "Phikon"]
                        if (OUT / f"{a}__a100.npy").exists()]
    rows = []
    for arm in arms:
        X100, _, tf = arm_features(arm)
        pipe = fit_probe(X100, 1.0, 0, None, tf)
        for cond in ["orig"] + list(CONDS):
            X, tfx = feats_braco(arm, cond, sub)
            if X is None: continue
            yp = predict(pipe, X, tfx)
            r, cm = recalls_pt(ysub, yp)
            rows.append(dict(arm=arm, cond=cond, ba9=round(r.mean(), 2),
                             ba8=round(np.delete(r, 1).mean(), 2), rec_BACK=round(r[1], 2),
                             BACK_para_ADI=round(100 * cm[1, 0] / cm[1].sum(), 2),
                             rec_ADI=round(r[0], 2), rec_STR=round(r[7], 2)))
        print("ok", arm, flush=True)
    R = pd.DataFrame(rows)
    base = R[R.cond == "orig"].set_index("arm")
    R["queda9"] = R.apply(lambda x: round(base.loc[x.arm, "ba9"] - x.ba9, 2), axis=1)
    R["queda8"] = R.apply(lambda x: round(base.loc[x.arm, "ba8"] - x.ba8, 2), axis=1)
    R.to_csv(OUT / "B1_1_resultado.csv", index=False)
    q = R.pivot(index="arm", columns="cond", values="queda9")
    dif = (q.loc["LBP-COR-H"] - q.loc["LBP"]).round(2)
    txt = ["# B1.1 - desvio de cor sintetico no interno (subamostra 500/classe)\n",
           R.to_markdown(index=False), "\n\nQueda diferencial LBP-COL - LBP (9 classes) por condicao:\n",
           dif.to_frame("queda_dif").to_markdown(),
           "\n\nReferencia externa (Tab. II): 17,66 - 7,96 = 9,70 pontos. Regra: >= 4,85 -> 'basta'; "
           "< 2,4 -> 'nao explica'; entre -> 'explica parte'."]
    (OUT / "B1_1_resumo.md").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "extrai"
    if modo == "extrai":
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        extrai(int(os.environ.get("N_WORKERS", 12)))
    else:
        avalia()
