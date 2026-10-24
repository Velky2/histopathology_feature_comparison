# ============================================================================
# camera_ready/scripts/common.py
# Aparato compartilhado das analises pos-aceite (WVC 2026, camera-ready).
#
# Tudo aqui e copia FIEL do que gerou os numeros do paper (wvc_semana5_6.ipynb e
# wvc_semana7_reanalise.ipynb): mesma ordem canonica, mesmo split interno, mesma
# subamostragem de rotulo, mesmo probe, mesmo bootstrap. Nao "melhore" nada aqui:
# qualquer mudanca quebra a comparabilidade com as Tabelas I-V.
# ============================================================================
import os
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("WVC_ROOT", Path(__file__).resolve().parents[2]))
OUT56 = ROOT / "out_semana56"
PRED_DIR = OUT56 / "preds_7k"
CR_OUT = ROOT / "camera_ready" / "out"
CR_OUT.mkdir(parents=True, exist_ok=True)
CR_PRED = CR_OUT / "preds_posaceite"          # predicoes NOVAS (pos-aceite) ficam separadas
CR_PRED.mkdir(parents=True, exist_ok=True)

CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
NC = len(CLASSES)
# Contagens por classe registradas nas execucoes do HPC (ordem canonica = pastas
# ordenadas por classe, arquivos ordenados por nome). Reconstroi y_100k sem precisar
# das imagens.
COUNTS_100K = [10407, 10566, 11512, 11557, 8896, 13536, 8763, 10446, 14317]
COUNTS_7K = [1338, 847, 339, 634, 1035, 592, 741, 421, 1233]

INTERNAL_SPLIT_SEED = 20260816
INTERNAL_TEST_SIZE = 0.20
LABEL_FRACTIONS = [0.01, 0.05, 0.10, 0.25, 1.00]
LABEL_SEEDS = [0, 1, 2, 3, 4]
COMMON_D = 32
GLOBAL_SEED_56 = 20260818       # semente do bootstrap da Semana 5-6 / 7
PROBE_MAX_ITER = 2000

H4_MORFO = ["TUM", "NORM"]
H4_TEXTURAL = ["STR", "MUS", "ADI", "MUC"]
H4_NAO_PREREG = ["BACK", "DEB", "LYM"]

y_100k = np.repeat(np.arange(NC), COUNTS_100K).astype(np.int64)
_y7 = OUT56 / "semana7" / "y_7k.npy"
y_7k = np.load(_y7) if _y7.exists() else np.repeat(np.arange(NC), COUNTS_7K).astype(np.int64)
assert np.array_equal(np.bincount(y_7k, minlength=NC), COUNTS_7K)

# ---------------------------------------------------------------------------
# Split interno e subamostragem (identicos ao notebook)
# ---------------------------------------------------------------------------
from sklearn.model_selection import train_test_split

_idx_all = np.arange(len(y_100k))
idx_train_pool, idx_eval_int = train_test_split(
    _idx_all, test_size=INTERNAL_TEST_SIZE, stratify=y_100k, random_state=INTERNAL_SPLIT_SEED)
idx_train_pool = np.sort(idx_train_pool)
idx_eval_int = np.sort(idx_eval_int)
y_train_pool = y_100k[idx_train_pool]
y_int = y_100k[idx_eval_int]


def label_subsample(frac, seed):
    if frac >= 1.0:
        return np.arange(len(idx_train_pool))
    sub, _ = train_test_split(np.arange(len(idx_train_pool)), train_size=frac,
                              stratify=y_train_pool, random_state=seed)
    return np.sort(sub)


def seeds_for(frac):
    return LABEL_SEEDS if frac < 1.0 else LABEL_SEEDS[:1]


# ---------------------------------------------------------------------------
# Bracos: (features 100k, features 7k, transformacao) - igual a PAIRS da Semana 5-6
# ---------------------------------------------------------------------------
def hellinger(X):
    return np.sqrt(np.clip(np.asarray(X, dtype=np.float32), 0.0, None))


def _load(rel):
    return np.load(ROOT / rel, mmap_mode="r")


CACHE = {
    "LBP54":      ("cache_semana2/lbp_100k_n100000_d4e34ccc5e.npy", "cache_semana56/LBP54_7k_n7180_v1.npy"),
    "LBPHF":      ("cache_semana56/LBPHF_100k_n100000_v1.npy", "cache_semana56/LBPHF_7k_n7180_v1.npy"),
    "LBPCOR":     ("cache_semana56/LBPCOR_100k_n100000_v1o.npy", "cache_semana56/LBPCOR_7k_n7180_v1o.npy"),
    "FRACTALBM":  ("cache_semana56/FRACTALBM_100k_n100000_v1.npy", "cache_semana56/FRACTALBM_7k_n7180_v1.npy"),
    # O braco "Fractal" usa a reimplementacao FRACTALV1 nos dois lados (o cache da Semana 2
    # nao reproduzia a reimplementacao; ver celula 11.2). Conferido por regressao abaixo.
    "FRACTALV1":  ("cache_semana56/FRACTALV1_100k_n100000_clB_frac_v1r.npy",
                   "cache_semana56/FRACTALV1_7k_n7180_clB_frac_v1r.npy"),
    "PHIKON":     ("cache_semana2/phikon_100k_n100000_4d501ea942.npy", "cache_semana56/phikon_7k_n7180.npy"),
    "SIMCLR_s0_full": ("cache_semana34/simclr_s0_full_100k_n100000_7611ff644c.npy",
                       "cache_semana56/simclr_s0_full_7k_n7180.npy"),
    "SIMCLR_s1_full": ("cache_semana34/simclr_s1_full_100k_n100000_da152444f2.npy",
                       "cache_semana56/simclr_s1_full_7k_n7180.npy"),
    "SIMCLR_s2_full": ("cache_semana34/simclr_s2_full_100k_n100000_d1c41936c9.npy",
                       "cache_semana56/simclr_s2_full_7k_n7180.npy"),
    "SIMCLR_s0_abl": ("cache_semana34/simclr_s0_ablated_no_scale_color_100k_n100000_601250d162.npy",
                      "cache_semana56/simclr_s0_ablated_no_scale_color_7k_n7180.npy"),
    "SIMCLR_s1_abl": ("cache_semana34/simclr_s1_ablated_no_scale_color_100k_n100000_34bf7360be.npy",
                      "cache_semana56/simclr_s1_ablated_no_scale_color_7k_n7180.npy"),
}

ARMS = ["Fractal", "Fractal-BM", "LBP", "LBP-H", "LBP-HF-H", "LBP-COR-H", "LBP-ENR", "CLASSICO-MAX",
        "SimCLR_abl_s0", "SimCLR_abl_s1", "SimCLR_s0", "SimCLR_s1", "SimCLR_s2", "Phikon"]
CLASSICOS = ARMS[:8]
NOME_PAPER = {"Fractal": "FRAC", "Fractal-BM": "FRAC-BM", "LBP": "LBP", "LBP-H": "LBP-H",
              "LBP-HF-H": "LBP-HF", "LBP-COR-H": "LBP-COL", "LBP-ENR": "LBP-ENR",
              "CLASSICO-MAX": "CLASSIC-MAX", "SimCLR_abl_s0": "SimCLR-abl", "SimCLR_abl_s1": "SimCLR-abl s1",
              "SimCLR_s0": "SimCLR", "SimCLR_s1": "SimCLR s1", "SimCLR_s2": "SimCLR s2", "Phikon": "Phikon"}


def _concat(*Xs):
    return np.concatenate([np.asarray(X, dtype=np.float32) for X in Xs], axis=1)


def arm_features(arm, overrides=None):
    """Devolve (X100, X7, transform). `overrides` permite trocar o lado 7k (ou o 100k) por
    features recalculadas numa intervencao: dict desc -> (X100 ou None, X7 ou None)."""
    ov = overrides or {}

    def get(desc):
        a, b = CACHE[desc]
        X100, X7 = _load(a), _load(b)
        if desc in ov:
            o100, o7 = ov[desc]
            X100 = X100 if o100 is None else o100
            X7 = X7 if o7 is None else o7
        return X100, X7

    if arm in ("LBP", "LBP-H"):
        X100, X7 = get("LBP54"); return X100, X7, (hellinger if arm == "LBP-H" else None)
    if arm == "LBP-HF-H":
        X100, X7 = get("LBPHF"); return X100, X7, hellinger
    if arm == "LBP-COR-H":
        X100, X7 = get("LBPCOR"); return X100, X7, hellinger
    if arm in ("LBP-ENR", "CLASSICO-MAX"):
        hf100, hf7 = get("LBPHF"); co100, co7 = get("LBPCOR")
        A100, A7 = _concat(hellinger(hf100), hellinger(co100)), _concat(hellinger(hf7), hellinger(co7))
        if arm == "CLASSICO-MAX":
            bm100, bm7 = get("FRACTALBM")
            A100, A7 = _concat(A100, bm100), _concat(A7, bm7)
        return A100, A7, None
    if arm == "Fractal":
        X100, X7 = get("FRACTALV1"); return X100, X7, None
    if arm == "Fractal-BM":
        X100, X7 = get("FRACTALBM"); return X100, X7, None
    if arm == "Phikon":
        X100, X7 = get("PHIKON"); return X100, X7, None
    if arm.startswith("SimCLR_abl_s"):
        X100, X7 = get(f"SIMCLR_s{arm[-1]}_abl"); return X100, X7, None
    if arm.startswith("SimCLR_s"):
        X100, X7 = get(f"SIMCLR_s{arm[-1]}_full"); return X100, X7, None
    raise KeyError(arm)


# ---------------------------------------------------------------------------
# Probe (identico): StandardScaler -> [PCA(d) -> StandardScaler] -> LogReg
# ---------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def make_probe(pca_dim=None):
    steps = [("sc", StandardScaler())]
    if pca_dim is not None:
        steps += [("pca", PCA(n_components=pca_dim, random_state=0)), ("sc2", StandardScaler())]
    steps += [("clf", LogisticRegression(max_iter=PROBE_MAX_ITER, class_weight="balanced", n_jobs=1))]
    return Pipeline(steps)


def fit_probe(X100, frac, seed, pca_dim=None, transform=None):
    tr = idx_train_pool[label_subsample(frac, seed)]
    Xtr = np.asarray(X100[tr])
    if transform is not None:
        Xtr = transform(Xtr)
    pipe = make_probe(pca_dim)
    pipe.fit(Xtr, y_100k[tr])
    return pipe


def predict(pipe, X, transform=None):
    X = np.asarray(X)
    if transform is not None:
        X = transform(X)
    return pipe.predict(X)


def recalls_pt(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=range(NC))
    return 100.0 * np.diag(cm) / np.maximum(cm.sum(1), 1), cm


def bal_acc(y_true, y_pred, classes=None):
    r, _ = recalls_pt(y_true, y_pred)
    if classes is not None:
        r = r[[CLASSES.index(c) for c in classes]]
    return float(r.mean())


def pred_path(arm, cond, frac, seed):
    return PRED_DIR / f"{arm}__{cond}__f{int(round(frac * 100))}__s{seed}.npy"


def load_pred(arm, cond, frac, seed):
    return np.load(pred_path(arm, cond, frac, seed)).astype(np.int64)


# ---------------------------------------------------------------------------
# Bootstrap estratificado por classe (mesmo gerador e ordem de sorteio da Semana 5-6)
# ---------------------------------------------------------------------------
POS7 = [np.where(y_7k == c)[0] for c in range(NC)]
NPC7 = np.array([len(p) for p in POS7])


def build_boot_local(B, seed, pos=POS7):
    """Indices LOCAIS (dentro de cada classe), sorteados na MESMA ordem do notebook original:
    para cada replica, para cada classe, rng.choice(pos_c, n_c, replace=True)."""
    rng = np.random.default_rng(seed)
    npc = [len(p) for p in pos]
    out = [np.empty((B, n), dtype=np.int32) for n in npc]
    for b in range(B):
        for c in range(len(pos)):
            # pos[c] e ordenado (np.where), entao searchsorted devolve o indice local exato
            out[c][b] = np.searchsorted(pos[c], rng.choice(pos[c], size=npc[c], replace=True))
    return out


def recall_boot(y_pred, local, pos=POS7, y_true=None):
    """(recall ponto [NC], recall bootstrap [B, NC]) em %."""
    y_true = y_7k if y_true is None else y_true
    ok = (y_pred == y_true)
    B = local[0].shape[0]
    r_pt = np.empty(NC); r_bt = np.empty((B, NC))
    for c in range(NC):
        okc = ok[pos[c]]
        r_pt[c] = 100.0 * okc.mean()
        r_bt[:, c] = 100.0 * okc[local[c]].mean(axis=1)
    return r_pt, r_bt


def holm(p):
    p = np.asarray(p, dtype=float); m = len(p); order = np.argsort(p, kind="stable")
    adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i]); adj[i] = min(run, 1.0)
    return adj


def tci(sd, n, conf=0.95):
    """Meia-largura do IC t de Student para a media de n valores com DP amostral sd."""
    from scipy import stats
    return float(stats.t.ppf(0.5 + conf / 2, n - 1) * sd / np.sqrt(n))
