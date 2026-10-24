# ============================================================================
# B0.4 (lado interno) + B1.5 (proxy de vazamento) + verificacao da Tab. II
#
#  1. Reajusta o probe de 100% (RAW, seed 0) de TODOS os bracos no pool de 80k e
#     prediz o split interno (20k) e o 7k. O 7k sai so para conferir que o refit
#     reproduz preds_7k/ (nenhum numero novo do 7k e usado daqui).
#  2. Acuracia balanceada de 9 e de 8 classes (sem BACK) no interno -> Tab. II sem BACK.
#  3. Proxy de vazamento por vizinho mais proximo (cosseno) no espaco de features:
#     interno->pool vs 7k->pool, quase-duplicatas, duplicatas exatas por hash de pixel
#     (memmap do 100k) e acuracia interna por faixa de similaridade.
#
# Rodar:  python b1_interno_refit.py   (CPU; ~20-40 min com 4 processos)
# ============================================================================
import os, sys, time, hashlib, json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

N_PROC = int(os.environ.get("N_PROC", 4))
OUT = None

def _fit_one(arm):
    os.environ.setdefault("OMP_NUM_THREADS", "3")
    from common import (arm_features, fit_probe, predict, idx_eval_int, y_int, y_7k,
                        bal_acc, recalls_pt, load_pred, CR_OUT)
    t = time.time()
    X100, X7, tf = arm_features(arm)
    p = fit_probe(X100, 1.0, 0, None, tf)
    yi = predict(p, np.asarray(X100[idx_eval_int]), tf)
    y7 = predict(p, X7, tf)
    out = CR_OUT / "interno"; out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{arm}__RAW__f100__s0__interno.npy", yi.astype(np.int8))
    ref = load_pred(arm, "RAW", 1.0, 0)
    r_int, _ = recalls_pt(y_int, yi)
    import sklearn
    return dict(arm=arm, n_iter=int(p.named_steps["clf"].n_iter_[0]), sklearn=sklearn.__version__,
                agree_7k_publicado=float((y7 == ref).mean()),
                ba7_refit=bal_acc(y_7k, y7), ba7_publicado=bal_acc(y_7k, ref),
                **{f"int_{c}": float(v) for c, v in zip(
                    ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"], r_int)},
                segundos=round(time.time() - t))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import *
    OUT = CR_OUT / "interno"; OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---------------- 1-2. refit interno ----------------
    fp = OUT / "refit_interno_100.csv"
    if fp.exists():
        R = pd.read_csv(fp)
    else:
        R = pd.DataFrame(Parallel(n_jobs=N_PROC, verbose=5)(delayed(_fit_one)(a) for a in ARMS))
        R.to_csv(fp, index=False)
    print(f"refit ok ({time.time()-t0:.0f}s)")

    pub_int = pd.read_csv(OUT56 / "h3b_escada_classica_interno.csv")
    pub_int = pub_int[pub_int.frac == 1.0].set_index("arm").bal_acc.to_dict()
    pub_int.update({"Phikon": 99.80, "SimCLR_s0": 97.96})
    rec_pub = pd.read_csv(OUT56 / "h4_recall_interno.csv", index_col=0)   # LBP, LBP-COR-H, Phikon, SimCLR(s0)
    rec_pub = rec_pub.rename(columns={"SimCLR": "SimCLR_s0"})
    SEM_BACK = [c for c in CLASSES if c != "BACK"]
    # externo SEM arredondar, direto das predicoes publicadas (100%, seed 0)
    ext = {}
    for a in ARMS:
        r7, _ = recalls_pt(y_7k, load_pred(a, "RAW", 1.0, 0))
        ext[a] = dict(ba9=r7.mean(), ba8_semBACK=np.delete(r7, 1).mean(), recall_BACK=r7[1])
    ext = pd.DataFrame(ext).T

    rows = []
    for _, r in R.iterrows():
        rv = np.array([r[f"int_{c}"] for c in CLASSES])
        fonte = "refit (sklearn %s)" % r.get("sklearn", "1.6.1, local")
        if r.arm in rec_pub.columns:                     # usa o recall interno PUBLICADO quando existe
            rv = rec_pub[r.arm].reindex(CLASSES).values.astype(float); fonte = "publicado (h4_recall_interno.csv)"
        rows.append(dict(arm=r.arm, fonte_interno=fonte,
                         int9=rv.mean(), int9_publicado=pub_int.get(r.arm, np.nan),
                         int8=rv[[CLASSES.index(c) for c in SEM_BACK]].mean(),
                         int_BACK=rv[1],
                         ext9=ext.loc[r.arm, "ba9"], ext8=ext.loc[r.arm, "ba8_semBACK"],
                         ext_BACK=ext.loc[r.arm, "recall_BACK"],
                         agree_7k=round(r.agree_7k_publicado, 4), n_iter=r.n_iter))
    T = pd.DataFrame(rows).set_index("arm")
    for k in ["9", "8"]:
        T[f"queda{k}"] = (T[f"int{k}"] - T[f"ext{k}"]).round(2)
        for L, nome in [("Phikon", "Phi"), ("SimCLR_s0", "SSL")]:
            T[f"ampl{nome}{k}"] = ((T.loc[L, f"int{k}"] - T[f"int{k}"]) * -1 + (T.loc[L, f"ext{k}"] - T[f"ext{k}"])).round(2)
    T = T.round(2)
    T.to_csv(OUT / "tabela2_9_e_8_classes.csv")
    g_int9 = T.loc["LBP-COR-H", "int9"] - T.loc["LBP", "int9"]; g_ext9 = T.loc["LBP-COR-H", "ext9"] - T.loc["LBP", "ext9"]
    g_int8 = T.loc["LBP-COR-H", "int8"] - T.loc["LBP", "int8"]; g_ext8 = T.loc["LBP-COR-H", "ext8"] - T.loc["LBP", "ext8"]
    resumo = dict(ganho_cor_int9=round(g_int9, 2), ganho_cor_ext9=round(g_ext9, 2),
                  fracao_que_transfere_9=round(g_ext9 / g_int9, 3),
                  ganho_cor_int8=round(g_int8, 2), ganho_cor_ext8=round(g_ext8, 2),
                  fracao_que_transfere_8=round(g_ext8 / g_int8, 3))
    print(T.to_string()); print(resumo)

    # ---------------- 3. proxy de vazamento ----------------
    def l2n(X):
        X = np.asarray(X, dtype=np.float32)
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    def nn_sim(Q, P, chunk=2048):
        best = np.empty(len(Q), np.float32); arg = np.empty(len(Q), np.int64)
        for i in range(0, len(Q), chunk):
            S = Q[i:i + chunk] @ P.T
            arg[i:i + chunk] = S.argmax(1); best[i:i + chunk] = S.max(1)
        return best, arg

    leak_rows, bins_rows = [], []
    pool_y = y_100k[idx_train_pool]
    for arm, desc_tf in [("Phikon", None), ("SimCLR_s0", None), ("LBP-COR-H", "std")]:
        X100, X7, tf = arm_features(arm)
        A = np.asarray(X100, dtype=np.float32); B7 = np.asarray(X7, dtype=np.float32)
        if tf is not None: A, B7 = tf(A), tf(B7)
        if desc_tf == "std":      # histogramas: cosseno apos padronizacao (mesma geometria do probe)
            mu, sd = A[idx_train_pool].mean(0), A[idx_train_pool].std(0) + 1e-6
            A, B7 = (A - mu) / sd, (B7 - mu) / sd
        P = l2n(A[idx_train_pool]); Qi = l2n(A[idx_eval_int]); Q7 = l2n(B7)
        si, ai = nn_sim(Qi, P); s7, a7 = nn_sim(Q7, P)
        thr99 = float(np.percentile(s7, 99)); thr_max = float(s7.max())
        yi_pred = np.load(OUT / f"{arm}__RAW__f100__s0__interno.npy").astype(int)
        if arm in ("Phikon", "SimCLR_s0", "LBP-COR-H"):
            pass
        leak_rows.append(dict(espaco=arm,
                              sim_mediana_interno=round(float(np.median(si)), 4),
                              sim_mediana_7k=round(float(np.median(s7)), 4),
                              sim_p01_7k=round(float(np.percentile(s7, 1)), 4),
                              sim_p99_7k=round(thr99, 4), sim_max_7k=round(thr_max, 4),
                              frac_interno_acima_p99_7k=round(float((si > thr99).mean()), 4),
                              frac_interno_acima_max_7k=round(float((si > thr_max).mean()), 4),
                              nn_mesma_classe_interno=round(float((pool_y[ai] == y_int).mean()), 4),
                              nn_mesma_classe_7k=round(float((pool_y[a7] == y_7k).mean()), 4),
                              dup_feature_interno=int((si > 1 - 1e-6).sum()),
                              dup_feature_7k=int((s7 > 1 - 1e-6).sum())))
        np.save(OUT / f"B1_5_sim_{arm}_interno.npy", si); np.save(OUT / f"B1_5_sim_{arm}_7k.npy", s7)
        # (i) quintis globais: acuracia SIMPLES e BALANCEADA; (ii) quintis DENTRO de cada classe
        qs = np.quantile(si, [0, .2, .4, .6, .8, 1.0])
        rank_cls = np.empty(len(si), dtype=int)
        for c in range(NC):
            mc = (y_int == c); r = np.argsort(np.argsort(si[mc]))
            rank_cls[mc] = np.minimum((5 * r) // mc.sum(), 4)
        for k in range(5):
            m = (si >= qs[k]) & (si <= qs[k + 1]) if k == 4 else (si >= qs[k]) & (si < qs[k + 1])
            mw = rank_cls == k
            bins_rows.append(dict(espaco=arm, faixa=f"Q{k+1}", sim_lo=round(float(qs[k]), 4),
                                  sim_hi=round(float(qs[k + 1]), 4), n=int(m.sum()),
                                  acc_simples=round(100 * float((yi_pred[m] == y_int[m]).mean()), 2),
                                  ba_global=round(bal_acc(y_int[m], yi_pred[m]), 2),
                                  ba_quintil_dentro_da_classe=round(bal_acc(y_int[mw], yi_pred[mw]), 2),
                                  frac_acima_p99_7k=round(float((si[m] > thr99).mean()), 3)))
        y7p = load_pred(arm, "RAW", 1.0, 0)
        bins_rows.append(dict(espaco=arm, faixa="7k", n=len(y7p), acc_simples=round(100 * float((y7p == y_7k).mean()), 2),
                              ba_global=round(bal_acc(y_7k, y7p), 2), ba_quintil_dentro_da_classe=np.nan))
        print(f"vazamento {arm} ok ({time.time()-t0:.0f}s)")
    L = pd.DataFrame(leak_rows); L.to_csv(OUT / "B1_5_vazamento_nn.csv", index=False)
    Bn = pd.DataFrame(bins_rows); Bn.to_csv(OUT / "B1_5_acc_por_faixa.csv", index=False)

    # duplicatas EXATAS de pixel dentro do 100k (pool x interno), pelo memmap
    mm_path = ROOT / "nct100k_u8.mmap"
    dup = {}
    if mm_path.exists():
        mm = np.memmap(mm_path, dtype=np.uint8, mode="r", shape=(100000, 224, 224, 3))
        h = np.empty(100000, dtype="S16")
        for i in range(100000):
            h[i] = hashlib.md5(mm[i].tobytes()).digest()
        hp = set(h[idx_train_pool].tolist())
        n_int_dup = int(sum(x in hp for x in h[idx_eval_int].tolist()))
        n_unique = len(set(h.tolist()))
        dup = dict(duplicatas_exatas_interno_no_pool=n_int_dup, hashes_unicos_100k=n_unique)
        print("duplicatas exatas:", dup)
    json.dump(dict(resumo_cor=resumo, duplicatas=dup), open(OUT / "B0_4_B1_5_resumo.json", "w"), indent=2)
    print(L.to_string()); print(Bn.to_string())
    print(f"fim ({time.time()-t0:.0f}s)")
