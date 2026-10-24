# ============================================================================
# Transformacoes de cor das analises B1.1 (desvio sintetico no interno) e B1.2
# (renormalizacao global do 7k). Deterministicas, globais e independentes do rotulo.
# Usado tanto pelos scripts de CPU (descritores classicos) quanto pelo de GPU (SimCLR/Phikon),
# para que os dois lados vejam EXATAMENTE os mesmos pixels.
# ============================================================================
import json
import numpy as np
from common import OUT56, idx_eval_int, y_100k, NC

_D = json.loads((OUT56 / "diagnostico_stain.json").read_text())
MU100 = np.array(_D["od_100k"], dtype=np.float64)          # OD media por imagem, media da coorte
MU7 = np.array(_D["od_7k"], dtype=np.float64)
DELTA = MU7 - MU100                                          # [0.0439, 0.0388, 0.0224]
RATIO = np.array(_D["sd_7k"]) / np.array(_D["sd_100k"])    # [1.341, 1.030, 1.266]

CONDS = {               # nome -> (alpha, inflar_dispersao)
    "a050": (0.5, False),
    "a100": (1.0, False),
    "a200": (2.0, False),
    "a100_disp": (1.0, True),
    "ajcomp": (1.0, False),     # POST HOC (errata do adendo): DELTA ajustado pela composicao de classes
}
# DELTA ajustado por composicao (post hoc, 22/10): MU7 - [MU100 + (w7 - w100) . OD_medio_por_classe(100k)],
# OD por classe medida em 300 patches/classe do pool (semente 20261025). A diferenca entre as medias
# das coortes mistura cor com composicao (o 7k tem outra proporcao de classes); no canal G o desvio
# por classe e ~2x o DELTA bruto.
DELTA_AJCOMP = np.array([0.0489, 0.0792, 0.0376])
SUB_SEED = 20261022
SUB_PER_CLASS = 500


def od(rgb_u8):
    return -np.log(np.clip(rgb_u8.astype(np.float64) / 255.0, 1e-3, 1.0))


def from_od(OD):
    return np.clip(np.rint(255.0 * np.exp(-OD)), 0, 255).astype(np.uint8)


def desvio_cor(rgb_u8, alpha, disp, delta=None):
    """B1.1: OD' = OD + alpha*DELTA [+ (RATIO-1)*(m_i - MU100)], m_i = OD medio da imagem."""
    O = od(rgb_u8)
    shift = alpha * (DELTA if delta is None else delta)
    if disp:
        m_i = O.reshape(-1, 3).mean(0)
        shift = shift + (RATIO - 1.0) * (m_i - MU100)
    return from_od(O + shift)


def aplica_cond(rgb_u8, cond):
    """Aplica a condicao `cond` de CONDS (usado igual nos lados CPU e GPU)."""
    alpha, disp = CONDS[cond]
    return desvio_cor(rgb_u8, alpha, disp, delta=DELTA_AJCOMP if cond == "ajcomp" else None)


def subamostra_interna():
    """500 patches/classe do split interno, sorteio fixo (indices globais do 100k, ordenados)."""
    rng = np.random.default_rng(SUB_SEED)
    yi = y_100k[idx_eval_int]
    sel = [rng.choice(idx_eval_int[yi == c], size=SUB_PER_CLASS, replace=False) for c in range(NC)]
    return np.sort(np.concatenate(sel))


# ---------------------------------------------------------------------------
# B1.2: renormalizacao global do 7k para as estatisticas de OD do pool de 80k
#   estatisticas em nivel de PIXEL, por canal, sobre 1000 imagens sorteadas de cada coorte
#   (semente fixa, sem olhar rotulo). OD'_7k = MU_pool + (OD_7k - MU_7k) * SD_pool / SD_7k
# ---------------------------------------------------------------------------
RENORM_SEED = 20261023
RENORM_N = 1000


def od_pixel_stats(get_rgb, n_total, n=RENORM_N, seed=RENORM_SEED, idx_pool=None):
    rng = np.random.default_rng(seed)
    cand = np.arange(n_total) if idx_pool is None else np.asarray(idx_pool)
    sel = rng.choice(cand, size=min(n, len(cand)), replace=False)
    s1 = np.zeros(3); s2 = np.zeros(3); cnt = 0
    for i in sel:
        O = od(get_rgb(int(i))).reshape(-1, 3)
        s1 += O.sum(0); s2 += (O * O).sum(0); cnt += O.shape[0]
    mu = s1 / cnt
    sd = np.sqrt(np.maximum(s2 / cnt - mu * mu, 1e-12))
    return mu, sd


def renormaliza(rgb_u8, mu_src, sd_src, mu_dst, sd_dst):
    O = od(rgb_u8)
    return from_od(mu_dst + (O - mu_src) * (sd_dst / sd_src))
