# ============================================================================
# B1.2 - 7k renormalizado para as estatisticas globais de OD do pool de 80k
# (pos-aceite, exploratoria; criterio fixo em adendo_posaceite.md). Precisa das imagens
# do CRC-VAL-HE-7K -> roda no HPC.
#
#   python b1_2_renorm_7k.py extrai   # descritores classicos no 7k renormalizado (CPU)
#   python b1_2_renorm_7k.py avalia   # probes de 100% (pool 80k) no 7k original e renormalizado
#                                      # inclui SimCLR/Phikon se extrai_gpu_camera_ready.py
#                                      # --modo renorm_7k ja tiver gravado as features
# ============================================================================
import os, sys, json, time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from intervencao_cor import od_pixel_stats, renormaliza

OUT = CR_OUT / "B1_2"; OUT.mkdir(parents=True, exist_ok=True)
CRC = Path(os.environ.get("CRC_VAL_DIR", ROOT / "CRC-VAL-HE-7K"))
DESCS = ["LBP54", "LBPHF", "LBPCOR", "FRACTALBM", "FRACTALV1"]


def paths_7k():
    ps = []
    for c in CLASSES:
        ps += sorted(p for p in (CRC / c).iterdir() if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"})
    assert len(ps) == 7180, len(ps)
    return ps


def stats():
    st = OUT / "od_stats.json"
    if st.exists():
        S = json.loads(st.read_text())
    else:
        mm = np.memmap(ROOT / "nct100k_u8.mmap", dtype=np.uint8, mode="r", shape=(100000, 224, 224, 3))
        p7 = paths_7k()
        mu_p, sd_p = od_pixel_stats(lambda i: np.asarray(mm[i]), 100000, idx_pool=idx_train_pool)
        mu_7, sd_7 = od_pixel_stats(lambda k: np.asarray(Image.open(p7[k]).convert("RGB")), len(p7))
        S = dict(mu_pool=mu_p.tolist(), sd_pool=sd_p.tolist(), mu_7k=mu_7.tolist(), sd_7k=sd_7.tolist())
        tmp = st.with_suffix(".tmp"); tmp.write_text(json.dumps(S, indent=2)); os.replace(tmp, st)
    return {k: np.array(v) for k, v in S.items()}


def _worker(args):
    from descritores import compute
    plist, S = args
    out = {d: [] for d in DESCS}
    for p in plist:
        a = renormaliza(np.asarray(Image.open(p).convert("RGB")), S["mu_7k"], S["sd_7k"], S["mu_pool"], S["sd_pool"])
        for d in DESCS:
            out[d].append(compute(d, a))
    return {d: np.asarray(v, dtype=np.float32) for d, v in out.items()}


def extrai(nw):
    S = stats(); print("OD stats:", {k: np.round(v, 4).tolist() for k, v in S.items()})
    ps = [str(p) for p in paths_7k()]
    from descritores import compute          # o recalculo do 7k ORIGINAL bate com o cache?
    for d in DESCS:
        ref = np.load(ROOT / CACHE[d][1], mmap_mode="r")[:8]
        got = np.array([compute(d, np.asarray(Image.open(p).convert("RGB"))) for p in ps[:8]])
        assert np.abs(got - ref).max() < 1e-4, f"{d}: recalculo do 7k diverge do cache"
    print("descritores do 7k original conferem com o cache (8 imagens)")
    if all((OUT / f"{d}__renorm.npy").exists() for d in DESCS):
        print("[cache] classicos renorm"); return
    t = time.time(); parts = {d: [] for d in DESCS}
    chunks = [(ps[i:i + 64], S) for i in range(0, len(ps), 64)]
    with ProcessPoolExecutor(nw) as ex:
        for k, r in enumerate(ex.map(_worker, chunks)):
            for d in DESCS: parts[d].append(r[d])
            if (k + 1) % 20 == 0: print(f"  {64*(k+1)}/{len(ps)} ({(time.time()-t)/60:.1f} min)", flush=True)
    for d in DESCS:
        np.save(OUT / f"{d}__renorm.npy", np.concatenate(parts[d]))
    print(f"[ok] classicos renorm em {(time.time()-t)/60:.1f} min")


def avalia():
    ov = {d: (None, np.load(OUT / f"{d}__renorm.npy")) for d in DESCS}
    ov_learn = {"SimCLR_s0": "SIMCLR_s0_full", "SimCLR_s1": "SIMCLR_s1_full", "SimCLR_s2": "SIMCLR_s2_full",
                "Phikon": "PHIKON"}
    rows = []
    for arm in ARMS:
        o = dict(ov)
        if arm in ov_learn:
            fp = OUT / f"{arm}__renorm.npy"
            if not fp.exists(): continue
            o[ov_learn[arm]] = (None, np.load(fp))
        elif arm.startswith("SimCLR_abl"):
            continue
        X100, X7, tf = arm_features(arm)
        _, X7r, _ = arm_features(arm, overrides=o)
        pipe = fit_probe(X100, 1.0, 0, None, tf)
        y_or, y_rn = predict(pipe, X7, tf), predict(pipe, X7r, tf)
        pub = load_pred(arm, "RAW", 1.0, 0)
        for nome, yp in [("original", y_or), ("renorm", y_rn)]:
            r, cm = recalls_pt(y_7k, yp)
            rows.append(dict(arm=arm, cond=nome, ba9=round(r.mean(), 2), ba8=round(np.delete(r, 1).mean(), 2),
                             rec_BACK=round(r[1], 2), BACK_para_ADI=round(100 * cm[1, 0] / cm[1].sum(), 2),
                             concord_publicado=round(float((yp == pub).mean()), 4) if nome == "original" else None))
        print("ok", arm, flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT / "B1_2_resultado.csv", index=False)
    P = R.pivot(index="arm", columns="cond", values="ba9")
    P["ganho"] = (P["renorm"] - P["original"]).round(2)
    B = R.pivot(index="arm", columns="cond", values="rec_BACK"); P["ganho_BACK"] = (B["renorm"] - B["original"]).round(2)
    txt = ["# B1.2 - 7k renormalizado (OD global do pool)\n", R.to_markdown(index=False), "\n\n", P.to_markdown(),
           "\n\nRegra: LBP-COL >= +3 pt e BACK >= +25 pt, com LBP e aprendidos |delta| <= 1 -> "
           "'as estatisticas globais de cor explicam boa parte da queda'; senao, 'nao se reduz a cor global'."]
    (OUT / "B1_2_resumo.md").write_text("\n".join(txt), encoding="utf-8"); print("\n".join(txt))


if __name__ == "__main__":
    if (sys.argv[1] if len(sys.argv) > 1 else "extrai") == "extrai":
        extrai(int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("N_WORKERS", 12))))
    else:
        avalia()
