# ============================================================================
# B1.3 - Proxies de estimabilidade/consistencia da normalizacao (R2.1, R4.3)
#   NAO e "taxa de sucesso": a normalizacao de Macenko foi feita pelos autores do dataset
#   e nao ha log. O que se mede, por coorte e classe:
#     (i)   fracao de pixels "de tecido" (OD >= 0,15 em todos os canais, limiar beta de Macenko);
#           patch degenerado se < 1% (a estimativa de Macenko nao tem pixels suficientes)
#     (ii)  vetores H/E estimados por Macenko (alpha = 1) e angulo contra a mediana da coorte
#     (iii) fracao de pixels saturados (255) e pretos (0) em algum canal
#     (iv)  indice de blocagem JPEG 8x8 (gradiente medio nas fronteiras de bloco / fora delas)
#     (v)   compressao declarada no cabecalho TIFF (amostra) e padrao de nome de arquivo do 7k
#
#   python b1_3_proxies_normalizacao.py [--so-100k]   (100k le o memmap; 7k precisa da pasta)
# ============================================================================
import os, sys, json, time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

OUT = CR_OUT / "B1_3"; OUT.mkdir(parents=True, exist_ok=True)
BETA, ALPHA, DEGEN = 0.15, 1.0, 0.01
N_POR_CLASSE_100K = 1000
CRC = Path(os.environ.get("CRC_VAL_DIR", ROOT / "CRC-VAL-HE-7K"))
NCT = Path(os.environ.get("NCT_DIR", ROOT / "NCT-CRC-HE-100K"))


def macenko_vectors(O):
    """O: (n,3) OD de pixels de tecido. Devolve (H, E) unitarios ou None."""
    if len(O) < 50: return None
    C = np.cov(O.T)
    w, V = np.linalg.eigh(C)
    V = V[:, [1, 2]]                                    # dois maiores autovetores
    if V[0, 0] < 0: V[:, 0] *= -1
    if V[0, 1] < 0: V[:, 1] *= -1
    T = O @ V
    phi = np.arctan2(T[:, 1], T[:, 0])
    a1, a2 = np.percentile(phi, ALPHA), np.percentile(phi, 100 - ALPHA)
    v1 = V @ np.array([np.cos(a1), np.sin(a1)]); v2 = V @ np.array([np.cos(a2), np.sin(a2)])
    H, E = (v1, v2) if v1[0] > v2[0] else (v2, v1)     # H tem mais OD no vermelho
    return H / np.linalg.norm(H), E / np.linalg.norm(E)


def blockiness(gray):
    g = gray.astype(np.float64)
    dh = np.abs(np.diff(g, axis=1)); dv = np.abs(np.diff(g, axis=0))
    cols = np.arange(dh.shape[1]); rows = np.arange(dv.shape[0])
    bh, bv = (cols % 8 == 7), (rows % 8 == 7)
    inner = (dh[:, ~bh].mean() + dv[~bv, :].mean()) / 2
    bound = (dh[:, bh].mean() + dv[bv, :].mean()) / 2
    return float(bound / max(inner, 1e-9))


def medir(rgb):
    O = -np.log(np.clip(rgb.reshape(-1, 3).astype(np.float64) / 255.0, 1e-3, 1.0))
    tecido = np.all(O >= BETA, axis=1)
    he = macenko_vectors(O[tecido]) if tecido.mean() >= DEGEN else None
    g = np.asarray(Image.fromarray(rgb).convert("L"))
    return dict(frac_tecido=float(tecido.mean()), degenerado=bool(tecido.mean() < DEGEN),
                H=None if he is None else he[0].tolist(), E=None if he is None else he[1].tolist(),
                sat255=float(np.any(rgb.reshape(-1, 3) == 255, axis=1).mean()),
                preto0=float(np.any(rgb.reshape(-1, 3) == 0, axis=1).mean()),
                bloco=blockiness(g))


def _w100(idxs):
    mm = np.memmap(ROOT / "nct100k_u8.mmap", dtype=np.uint8, mode="r", shape=(100000, 224, 224, 3))
    return [dict(i=int(i), **medir(np.asarray(mm[i]))) for i in idxs]


def _w7(plist):
    return [dict(i=int(k), **medir(np.asarray(Image.open(p).convert("RGB")))) for k, p in plist]


def resumo(D, coorte):
    Hm = np.median(np.array([h for h in D.H if h is not None]), axis=0); Hm /= np.linalg.norm(Hm)
    Em = np.median(np.array([e for e in D.E if e is not None]), axis=0); Em /= np.linalg.norm(Em)
    ang = lambda v, m: np.degrees(np.arccos(np.clip(np.dot(v, m), -1, 1))) if v is not None else np.nan
    D["ang_H"] = [ang(np.array(h) if h is not None else None, Hm) for h in D.H]
    D["ang_E"] = [ang(np.array(e) if e is not None else None, Em) for e in D.E]
    g = D.groupby("classe").agg(n=("i", "size"), frac_degenerado=("degenerado", "mean"),
                                frac_tecido_mediana=("frac_tecido", "median"),
                                ang_H_mediana=("ang_H", "median"), ang_E_mediana=("ang_E", "median"),
                                ang_H_p90=("ang_H", lambda x: np.nanpercentile(x, 90)),
                                sat255_media=("sat255", "mean"), bloco_mediana=("bloco", "median"))
    tot = D.agg(dict(degenerado="mean", frac_tecido="median", ang_H="median", ang_E="median",
                     sat255="mean", bloco="median"))
    g.loc["TODAS"] = [len(D), tot.degenerado, tot.frac_tecido, tot.ang_H, tot.ang_E,
                      np.nanpercentile(D.ang_H, 90), tot.sat255, tot.bloco]
    g.insert(0, "coorte", coorte)
    return g.round(4), Hm, Em


if __name__ == "__main__":
    so100 = "--so-100k" in sys.argv
    nw = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("N_WORKERS", 8)))
    t0 = time.time(); outs = []
    rng = np.random.default_rng(20261024)
    sel = np.sort(np.concatenate([rng.choice(idx_train_pool[y_train_pool == c], N_POR_CLASSE_100K, replace=False)
                                  for c in range(NC)]))
    with ProcessPoolExecutor(nw) as ex:
        R = [r for part in ex.map(_w100, [sel[i:i + 100] for i in range(0, len(sel), 100)]) for r in part]
    D100 = pd.DataFrame(R); D100["classe"] = [CLASSES[y_100k[i]] for i in D100.i]
    g100, H100, E100 = resumo(D100, "NCT-CRC-HE-100K (pool, 1000/classe)"); outs.append(g100)
    print(g100.to_string(), f"({time.time()-t0:.0f}s)", flush=True)
    info = dict(H_mediano_100k=H100.tolist(), E_mediano_100k=E100.tolist())
    if not so100:
        assert CRC.exists(), f"CRC_VAL_DIR nao encontrado: {CRC}"
        ps = []
        for c in CLASSES:
            ps += sorted(p for p in (CRC / c).iterdir() if p.suffix.lower() in {".tif", ".tiff"})
        assert len(ps) == 7180, len(ps)
        with ProcessPoolExecutor(nw) as ex:
            R = [r for part in ex.map(_w7, [list(enumerate(ps))[i:i + 100] for i in range(0, len(ps), 100)]) for r in part]
        D7 = pd.DataFrame(R).sort_values("i"); D7["classe"] = [CLASSES[y_7k[i]] for i in D7.i]
        g7, H7, E7 = resumo(D7, "CRC-VAL-HE-7K (todas)"); outs.append(g7)
        info.update(H_mediano_7k=H7.tolist(), E_mediano_7k=E7.tolist(),
                    angulo_H_100k_vs_7k=float(np.degrees(np.arccos(np.clip(H100 @ H7, -1, 1)))),
                    angulo_E_100k_vs_7k=float(np.degrees(np.arccos(np.clip(E100 @ E7, -1, 1)))))
        # cabecalho TIFF e nomes de arquivo (IDs de paciente?)
        comp = {}
        for nome, root in [("7k", CRC), ("100k", NCT)]:
            if not root.exists(): continue
            for c in CLASSES:
                fs = sorted((root / c).iterdir())[:20]
                comp[f"{nome}/{c}"] = dict(compressao=sorted({str(Image.open(f).info.get("compression")) for f in fs}),
                                           exemplos_nome=[f.name for f in fs[:3]])
        info["tiff"] = comp
        print(json.dumps(comp, indent=1))
    G = pd.concat(outs); G.to_csv(OUT / "B1_3_proxies.csv")
    (OUT / "B1_3_info.json").write_text(json.dumps(info, indent=2))
    (OUT / "B1_3_resumo.md").write_text("# B1.3 - proxies de normalizacao\n\n" + G.to_markdown() +
                                        "\n\n```json\n" + json.dumps(info, indent=2) + "\n```\n", encoding="utf-8")
    print(G.to_string()); print(json.dumps({k: v for k, v in info.items() if k != "tiff"}, indent=1))
    print(f"fim {time.time()-t0:.0f}s")
