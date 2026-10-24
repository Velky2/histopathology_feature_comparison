# ============================================================================
# Extracao de features de GPU para o camera-ready (SimCLR via checkpoint e Phikon).
# Pre-processamento IDENTICO ao que gerou os caches do paper (celulas 11.3/11.4 de
# wvc_semana5_6.ipynb); cada modo valida contra o cache antes de gravar (cos > 0,999).
#
#   --modo validar      : so a checagem de pre-processamento (32 imagens do 100k por braco)
#   --modo ckpt         : features 100k (memmap) + 7k (pasta) de um checkpoint novo (B2.1)
#                         --ckpt camera_ready/ckpt_2x2/simclr_s0_crop_semcor_best.pt --tag s0_crop_semcor
#   --modo cor_interno  : B1.1 - subamostra interna (500/classe) sob cada desvio de cor
#   --modo renorm_7k    : B1.2 - 7k renormalizado para as estatisticas de OD do pool
#   --bracos simclr_s0,phikon   (default; acrescente simclr_s1,simclr_s2 se quiser)
#   --device cpu        : teste local sem GPU
# ============================================================================
import os, sys, argparse, json, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ap = argparse.ArgumentParser()
ap.add_argument("--modo", required=True, choices=["validar", "ckpt", "cor_interno", "renorm_7k"])
ap.add_argument("--bracos", default="simclr_s0,phikon")
ap.add_argument("--ckpt", default=None)
ap.add_argument("--tag", default=None)
ap.add_argument("--crc7k", default=os.environ.get("CRC_VAL_DIR", "CRC-VAL-HE-7K"))
ap.add_argument("--device", default=None)
ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
ap.add_argument("--limite", type=int, default=0, help="so N imagens (teste)")
args = ap.parse_args()

import torch, torchvision
import torch.nn as nn
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from common import ROOT, CACHE, CLASSES, CR_OUT, y_100k, idx_train_pool
from intervencao_cor import (CONDS, aplica_cond, subamostra_interna, od_pixel_stats, renormaliza)

DEV = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
if DEV == "cuda":       # marca presenca na placa ja (o monitor do cluster cancela GPU ociosa)
    _keepalive = torch.zeros((2048, 2048), device="cuda")
print("torch", torch.__version__, "| device", DEV,
      torch.cuda.get_device_name(0) if DEV == "cuda" else "", flush=True)
MM = np.memmap(ROOT / "nct100k_u8.mmap", dtype=np.uint8, mode="r", shape=(100000, 224, 224, 3))
NORM_MEAN, NORM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
eval_tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize(NORM_MEAN, NORM_STD)])


def paths_7k():
    root = ROOT / args.crc7k if not os.path.isabs(args.crc7k) else Path(args.crc7k)
    paths = []
    for c in CLASSES:
        d = root / c
        paths += sorted(p for p in d.iterdir() if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"})
    assert len(paths) == 7180, f"7k com {len(paths)} imagens (esperado 7180) em {root}"
    return paths


# ------------------------- fontes de imagem (uint8 RGB) -------------------------
class Fonte(Dataset):
    """get(k) -> uint8 RGB; `fn` aplica a intervencao de cor (ou identidade)."""
    def __init__(self, getter, n, fn=None):
        self.getter, self.n, self.fn = getter, n, fn
    def __len__(self): return self.n
    def rgb(self, k):
        a = self.getter(k)
        return a if self.fn is None else self.fn(a)


class SimDS(Dataset):
    def __init__(self, fonte): self.f = fonte
    def __len__(self): return len(self.f)
    def __getitem__(self, k): return eval_tf(Image.fromarray(self.f.rgb(k)))


# ------------------------------- modelos -------------------------------
class SimCLRNet(nn.Module):
    def __init__(self, backbone_name="resnet50", proj_dim=128):
        super().__init__()
        base = torchvision.models.__dict__[backbone_name](weights=None)
        self.feat_dim = base.fc.in_features
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.stages = nn.Sequential(base.layer1, base.layer2, base.layer3, base.layer4)
        self.avgpool = base.avgpool
        self.grad_ckpt = False
        self.proj = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim), nn.ReLU(inplace=True),
                                  nn.Linear(self.feat_dim, proj_dim))
    def encode(self, x):
        return torch.flatten(self.avgpool(self.stages(self.stem(x))), 1)


def load_simclr(ckpt_path):
    ck = torch.load(ckpt_path, map_location=DEV, weights_only=False)
    m = SimCLRNet(ck.get("backbone", "resnet50"), 128).to(DEV)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


@torch.no_grad()
def feats_simclr(model, fonte, batch=128):
    dl = DataLoader(SimDS(fonte), batch_size=batch, shuffle=False, num_workers=args.workers, pin_memory=(DEV == "cuda"))
    return np.concatenate([model.encode(xb.to(DEV)).float().cpu().numpy() for xb in dl]).astype(np.float32)


_PHI = {}
def phikon():
    if not _PHI:
        from transformers import ViTModel, AutoImageProcessor
        _PHI["proc"] = AutoImageProcessor.from_pretrained("owkin/phikon")
        _PHI["mdl"] = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False).to(DEV).eval()
    return _PHI["proc"], _PHI["mdl"]


@torch.no_grad()
def feats_phikon(fonte, batch=64):
    proc, mdl = phikon()
    out = []
    for i in range(0, len(fonte), batch):
        ims = [Image.fromarray(fonte.rgb(k)) for k in range(i, min(i + batch, len(fonte)))]
        inp = proc(ims, return_tensors="pt").to(DEV)
        out.append(mdl(**inp).last_hidden_state[:, 0, :].float().cpu().numpy())
        if (i // batch) % 50 == 0: print(f"    phikon {i}/{len(fonte)}", flush=True)
    return np.concatenate(out).astype(np.float32)


CKPTS = {"simclr_s0": "ckpt_simclr/simclr_s0_full_best.pt", "simclr_s1": "ckpt_simclr/simclr_s1_full_best.pt",
         "simclr_s2": "ckpt_simclr/simclr_s2_full_best.pt"}
CACHEKEY = {"simclr_s0": "SIMCLR_s0_full", "simclr_s1": "SIMCLR_s1_full", "simclr_s2": "SIMCLR_s2_full",
            "phikon": "PHIKON"}
ARMNAME = {"simclr_s0": "SimCLR_s0", "simclr_s1": "SimCLR_s1", "simclr_s2": "SimCLR_s2", "phikon": "Phikon"}


def extrator(braco):
    if braco == "phikon":
        return lambda fonte: feats_phikon(fonte)
    m = load_simclr(ROOT / CKPTS[braco])
    return lambda fonte: feats_simclr(m, fonte)


def valida(braco, ext, n=32):
    """cos(recomputo, cache do 100k) em 32 imagens sorteadas como no notebook."""
    rs = np.random.default_rng(13 if braco == "phikon" else 17).choice(100000, size=n, replace=False)
    f = Fonte(lambda k: np.asarray(MM[rs[k]]), n)
    ref = ext(f); got = np.asarray(np.load(ROOT / CACHE[CACHEKEY[braco]][0], mmap_mode="r")[rs])
    cs = float(np.mean([np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12) for a, b in zip(ref, got)]))
    print(f"  [{braco}] cos(recomputo, cache 100k) = {cs:.5f}", flush=True)
    assert cs > 0.999, f"{braco}: pre-processamento diverge do cache - PARAR"


def salva(fp, X):
    """Gravacao atomica: um .npy truncado nunca fica com o nome final."""
    tmp = fp.with_name(fp.stem + ".tmp.npy"); np.save(tmp, X); os.replace(tmp, fp)


t0 = time.time()
bracos = [b.strip() for b in args.bracos.split(",") if b.strip()]

if args.modo == "validar":
    for b in bracos:
        valida(b, extrator(b))

elif args.modo == "cor_interno":
    OUT = CR_OUT / "B1_1"; OUT.mkdir(parents=True, exist_ok=True)
    sub = subamostra_interna()
    if args.limite: sub = sub[:args.limite]
    np.save(OUT / "subamostra_idx.npy", subamostra_interna())
    for b in bracos:
        ext = extrator(b); valida(b, ext)
        suf = "_TESTE" if args.limite else ""
        # a000 = linha de base RECALCULADA (sem desvio), para comparar recalculo contra recalculo
        for cond in ["a000"] + list(CONDS):
            fp = OUT / f"{ARMNAME[b]}__{cond}{suf}.npy"
            if fp.exists(): print("[cache]", fp.name); continue
            fn = None if cond == "a000" else (lambda a, c=cond: aplica_cond(a, c))
            X = ext(Fonte(lambda k: np.asarray(MM[sub[k]]), len(sub), fn=fn)); salva(fp, X)
            print(f"[ok] {fp.name} {X.shape} ({(time.time()-t0)/60:.1f} min)", flush=True)

elif args.modo == "renorm_7k":
    OUT = CR_OUT / "B1_2"; OUT.mkdir(parents=True, exist_ok=True)
    p7 = paths_7k()
    load7 = lambda k: np.asarray(Image.open(p7[k]).convert("RGB"))
    st = OUT / "od_stats.json"
    if st.exists():
        S = json.loads(st.read_text())
    else:
        mu_p, sd_p = od_pixel_stats(lambda i: np.asarray(MM[i]), 100000, idx_pool=idx_train_pool)
        mu_7, sd_7 = od_pixel_stats(load7, len(p7))
        S = dict(mu_pool=mu_p.tolist(), sd_pool=sd_p.tolist(), mu_7k=mu_7.tolist(), sd_7k=sd_7.tolist())
        tmp = st.with_suffix(".tmp"); tmp.write_text(json.dumps(S, indent=2)); os.replace(tmp, st)
    print("estatisticas OD:", S, flush=True)
    fn = lambda a: renormaliza(a, np.array(S["mu_7k"]), np.array(S["sd_7k"]), np.array(S["mu_pool"]), np.array(S["sd_pool"]))
    n = args.limite or len(p7)
    for b in bracos:
        ext = extrator(b); valida(b, ext)
        fp = OUT / f"{ARMNAME[b]}__renorm{'_TESTE' if args.limite else ''}.npy"
        if fp.exists(): print("[cache]", fp.name); continue
        X = ext(Fonte(load7, n, fn=fn)); salva(fp, X)
        print(f"[ok] {fp.name} {X.shape} ({(time.time()-t0)/60:.1f} min)", flush=True)
        if not args.limite:     # sanidade: sem renormalizacao reproduz o cache do 7k?
            X0 = ext(Fonte(load7, 64)); ref = np.load(ROOT / CACHE[CACHEKEY[b]][1], mmap_mode="r")[:64]
            cs = float(np.mean([np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c) + 1e-12) for a, c in zip(X0, ref)]))
            print(f"  [{b}] cos(7k original recomputado, cache 7k) = {cs:.5f}", flush=True)

elif args.modo == "ckpt":
    assert args.ckpt and args.tag, "--ckpt e --tag obrigatorios"
    OUT = ROOT / "camera_ready" / "cache_2x2"; OUT.mkdir(parents=True, exist_ok=True)
    m = load_simclr(ROOT / args.ckpt if not os.path.isabs(args.ckpt) else args.ckpt)
    n100 = args.limite or 100000
    suf = "_TESTE" if args.limite else ""
    fp = OUT / f"simclr_{args.tag}_100k{suf}.npy"
    if not fp.exists():
        X = feats_simclr(m, Fonte(lambda k: np.asarray(MM[k]), n100)); salva(fp, X)
        print(f"[ok] {fp.name} {X.shape} ({(time.time()-t0)/60:.1f} min)", flush=True)
    p7 = paths_7k()
    fp = OUT / f"simclr_{args.tag}_7k{suf}.npy"
    if not fp.exists():
        X = feats_simclr(m, Fonte(lambda k: np.asarray(Image.open(p7[k]).convert("RGB")), args.limite or len(p7)))
        salva(fp, X); print(f"[ok] {fp.name} {X.shape} ({(time.time()-t0)/60:.1f} min)", flush=True)

print(f"FIM {args.modo} em {(time.time()-t0)/60:.1f} min", flush=True)
