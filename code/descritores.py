# ============================================================================
# Descritores classicos - copia VERBATIM de wvc_semana5_6.ipynb (celulas 8 e 11.2).
# Nao editar: qualquer mudanca invalida o reuso dos probes/caches da Semana 5-6.
# Entradas: gray = uint8 (224,224) obtido por PIL .convert("L"); rgb = uint8 (224,224,3).
# ============================================================================
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from scipy.ndimage import uniform_filter, distance_transform_edt, map_coordinates

USE_OPPONENT_COLOR = True
BM_S, BM_Z = 64, 64

LBP_PR = [(8, 1), (16, 2), (24, 3)]
HF_PR = [(8, 1), (16, 2)]
OPP_PR = [(8, 1), (16, 2)]
OPP_PAIRS = [(0, 1), (1, 2), (2, 0)]


def lbp_riu2_hist(g, P, R):
    codes = local_binary_pattern(g, P, R, method="uniform")
    h, _ = np.histogram(codes.ravel(), bins=np.arange(P + 3), density=True)
    return h.astype(np.float32)


def desc_lbp54(gray):
    return np.concatenate([lbp_riu2_hist(gray, P, R) for P, R in LBP_PR])


_HF_LUT, _RIU2_LUT, _GRID = {}, {}, {}


def hf_lut(P):
    if P in _HF_LUT: return _HF_LUT[P]
    n = 1 << P
    codes = np.arange(n, dtype=np.uint32)
    bits = ((codes[:, None] >> np.arange(P, dtype=np.uint32)) & 1).astype(np.uint8)
    circ = np.concatenate([bits, bits[:, :1]], axis=1).astype(np.int16)
    trans = np.abs(np.diff(circ, axis=1)).sum(1)
    ones = bits.sum(1)
    rot = ((bits == 1) & (np.roll(bits, 1, axis=1) == 0)).argmax(1)
    G = (P - 1) * P
    lut = np.full(n, G + 2, dtype=np.int32)
    unif = trans <= 2
    m = unif & (ones > 0) & (ones < P)
    lut[m] = (ones[m] - 1) * P + rot[m]
    lut[unif & (ones == 0)] = G
    lut[unif & (ones == P)] = G + 1
    _HF_LUT[P] = lut
    return lut


def lbp_hf_block(g, P, R):
    codes = local_binary_pattern(g, P, R, method="default").astype(np.int64)
    cnt = np.bincount(hf_lut(P)[codes.ravel()], minlength=(P - 1) * P + 3).astype(np.float64)
    cnt /= max(cnt.sum(), 1e-12)
    H = cnt[:(P - 1) * P].reshape(P - 1, P)
    F = np.abs(np.fft.rfft(H, axis=1))[:, :P // 2 + 1]
    return np.concatenate([F.ravel(), cnt[(P - 1) * P:]]).astype(np.float32)


def desc_lbphf(gray):
    return np.concatenate([lbp_hf_block(gray, P, R) for P, R in HF_PR])


def riu2_lut(P):
    if P in _RIU2_LUT: return _RIU2_LUT[P]
    n = 1 << P
    codes = np.arange(n, dtype=np.uint32)
    bits = ((codes[:, None] >> np.arange(P, dtype=np.uint32)) & 1).astype(np.uint8)
    circ = np.concatenate([bits, bits[:, :1]], axis=1).astype(np.int16)
    trans = np.abs(np.diff(circ, axis=1)).sum(1)
    _RIU2_LUT[P] = np.where(trans <= 2, bits.sum(1), P + 1).astype(np.int32)
    return _RIU2_LUT[P]


def _grid(shape):
    if shape not in _GRID:
        _GRID[shape] = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    return _GRID[shape]


def sample_ring(img, P, R):
    yy, xx = _grid(img.shape)
    f = img.astype(np.float32)
    out = np.empty((P,) + img.shape, dtype=np.float32)
    for p in range(P):
        ang = 2.0 * np.pi * p / P
        out[p] = map_coordinates(f, [yy - R * np.sin(ang), xx + R * np.cos(ang)], order=1, mode="nearest")
    return out


def opponent_hist(chC, chN, P, R):
    nb = sample_ring(chN, P, R)
    ctr = chC.astype(np.float32)
    codes = np.zeros(ctr.shape, dtype=np.int64)
    for p in range(P):
        codes |= ((nb[p] >= ctr).astype(np.int64) << p)
    h = np.bincount(riu2_lut(P)[codes.ravel()], minlength=P + 2).astype(np.float32)
    return h / max(h.sum(), 1e-12)


def desc_lbpcolor(rgb):
    per = [lbp_riu2_hist(rgb[:, :, c], P, R) for c in range(3) for P, R in LBP_PR]
    if not USE_OPPONENT_COLOR:
        return np.concatenate(per)
    opp = [opponent_hist(rgb[:, :, a], rgb[:, :, b], P, R) for (a, b) in OPP_PAIRS for P, R in OPP_PR]
    return np.concatenate(per + opp)


BM_RADII = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28], dtype=np.float32)


def desc_bm(gray):
    g = np.asarray(Image.fromarray(gray).resize((BM_S, BM_S), Image.BILINEAR))
    z = (g.astype(np.int32) * (BM_Z - 1)) // 255
    vol = np.zeros((BM_S, BM_S, BM_Z), dtype=bool)
    xx, yy = np.meshgrid(np.arange(BM_S), np.arange(BM_S), indexing="ij")
    vol[xx.ravel(), yy.ravel(), z.ravel()] = True
    d = distance_transform_edt(~vol)
    V = np.array([float((d <= r).sum()) for r in BM_RADII])
    logV = np.log(np.maximum(V, 1.0)) / np.log(float(vol.size))
    m = BM_RADII <= 10
    slope = np.polyfit(np.log(BM_RADII[m]), np.log(np.maximum(V[m], 1.0)), 1)[0]
    return np.concatenate([logV, [3.0 - slope]]).astype(np.float32)


MF_BOXES = [2, 4, 7, 8, 14, 16, 28, 32, 56]
MF_Q = np.arange(-5.0, 5.01, 1.0)


def desc_mf(gray):
    Nn = gray.shape[0]
    I = gray.astype(np.float64) + 1.0
    tot = I.sum()
    logr, logZ = [], []
    for r in MF_BOXES:
        nb = Nn // r
        mu = I[:nb * r, :nb * r].reshape(nb, r, nb, r).sum(axis=(1, 3)) / tot
        mu = np.maximum(mu.ravel(), 1e-12)
        logZ.append(np.log((mu[None, :] ** MF_Q[:, None]).sum(axis=1)))
        logr.append(np.log(r / float(Nn)))
    tau = np.polyfit(np.array(logr), np.array(logZ), 1)[0]
    alpha = np.gradient(tau, MF_Q)
    f = MF_Q * alpha - tau
    a_min, a_max = alpha.min(), alpha.max()
    i0 = int(np.argmax(f)); a0 = alpha[i0]
    width = a_max - a_min
    asym = (a0 - a_min) / max(width, 1e-9)
    qi = {q: int(np.argmin(np.abs(MF_Q - q))) for q in (-5, 0, 5)}
    return np.array([a_min, a_max, width, a0, f[i0], asym, tau[qi[-5]], tau[qi[0]], tau[qi[5]]], dtype=np.float32)


LAC_R = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48]


def desc_lac(gray):
    I = gray.astype(np.float64)
    out = []
    for r in LAC_R:
        m = uniform_filter(I, size=r, mode="reflect")
        s = uniform_filter(I * I, size=r, mode="reflect")
        var = np.maximum(s - m * m, 0.0)
        mm = m.mean()
        out.append(np.log1p(var.mean() / max(mm * mm, 1e-9) + 1.0))
    return np.array(out, dtype=np.float32)


def desc_fractal_bm(gray):
    return np.concatenate([desc_bm(gray), desc_mf(gray), desc_lac(gray)])


# --- Fractal v1 (celula 11.2) ---
DBC_BOXES = [4, 8, 16, 28, 56]


def _dbc_dim(g, boxes=DBC_BOXES):
    Nn = g.shape[0]; I = g.astype(np.float64)
    xs, ys = [], []
    for s in boxes:
        nb = Nn // s
        blk = I[:nb * s, :nb * s].reshape(nb, s, nb, s)
        mx = blk.max(axis=(1, 3)); mn = blk.min(axis=(1, 3))
        h = max(s * 255.0 / Nn, 1e-9)
        nr = np.floor(mx / h) - np.ceil(mn / h) + 1
        xs.append(np.log(1.0 / s)); ys.append(np.log(max(nr.sum(), 1.0)))
    return float(np.polyfit(xs, ys, 1)[0])


def _fourier_slope(g):
    F = np.fft.fftshift(np.abs(np.fft.fft2(g.astype(np.float64))) ** 2)
    Nn = g.shape[0]; c = Nn // 2
    yy, xx = np.mgrid[0:Nn, 0:Nn]
    r = np.sqrt((yy - c) ** 2 + (xx - c) ** 2).astype(int)
    rmax = c
    prof = np.bincount(r.ravel(), F.ravel(), minlength=rmax + 1)[1:rmax]
    cnt = np.bincount(r.ravel(), minlength=rmax + 1)[1:rmax]
    prof = prof / np.maximum(cnt, 1)
    m = prof > 0
    a, b = np.polyfit(np.log(np.arange(1, rmax)[m]), np.log(prof[m]), 1)
    return float(a), float(b)


def _tiles(g, k):
    n = g.shape[0] // k
    return [g[i * n:(i + 1) * n, j * n:(j + 1) * n] for i in range(k) for j in range(k)]


def desc_fractal_v1(gray):
    if gray.std() < 1e-6:
        return np.zeros(38, dtype=np.float32)
    lac = desc_lac(gray)
    g_glob = _dbc_dim(gray)
    t2 = [_dbc_dim(t) for t in _tiles(gray, 2)]
    t4 = [_dbc_dim(t) for t in _tiles(gray, 4)]
    st = [np.mean(t4), np.std(t4), np.min(t4), np.max(t4), np.median(t4)]
    sl, ic = _fourier_slope(gray)
    return np.concatenate([lac, [g_glob], t2, t4, st, [sl, ic]]).astype(np.float32)


DESC = {"LBP54": (desc_lbp54, "L"), "LBPHF": (desc_lbphf, "L"), "LBPCOR": (desc_lbpcolor, "RGB"),
        "FRACTALBM": (desc_fractal_bm, "L"), "FRACTALV1": (desc_fractal_v1, "L")}


def compute(name, rgb_u8):
    fn, mode = DESC[name]
    if mode == "RGB":
        return fn(rgb_u8)
    return fn(np.asarray(Image.fromarray(rgb_u8).convert("L")))
