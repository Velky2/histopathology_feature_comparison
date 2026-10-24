# ============================================================================
# B2.1 - Ablacao fatorial 2x2 do SimCLR (camera-ready WVC 2026)
#
# Reproduz a RECEITA DA REVISAO 1 (a que gerou os bracos SimCLR e SimCLR-abl do paper),
# trocando apenas o conjunto de augmentations. Copia fiel de wvc_semana3_4_1.ipynb:
#   ResNet-50 (weights=None), projecao 2048-2048-128, NT-Xent tau=0.2, AdamW lr=1e-3 wd=1e-4,
#   cosseno com warm-up de 10 epocas e total_steps = 200 * len(dl) (agenda de 200 epocas
#   TRUNCADA, como no paper), AMP, batch 256, gradient checkpointing, channels_last,
#   pre-treino so no train_pool (80k), kNN-monitor (K=20, 40/classe treino + 40/classe aval,
#   sorteados de dentro do train_pool) a cada 5 epocas, checkpoint = melhor kNN-monitor.
#
# Unica diferenca de I/O: os pixels vem do memmap uint8 nct100k_u8.mmap (gerado a partir
# dos mesmos TIFF, Image.open().convert("RGB"), sem redimensionar) em vez do decode TIFF.
# Os transforms recebem o MESMO PIL.Image -> pipeline identico, so mais rapido.
#
# Orcamento: 20 epocas (= o que o timebox de 12 h deu a receita completa na T4: ep. 20).
# Tempo por epoca desconhecido na T4 com memmap (0,2-0,5 h); o log imprime a cada epoca.
# --wallclock-h e so rede de seguranca.
#
# Uso:
#   python simclr_2x2_treino.py --aug crop_semcor   --seed 0
#   python simclr_2x2_treino.py --aug semcrop_comcor --seed 0
#   (opcionais) --aug full_semgray | full_semrot | legacy_no_scale_color (reproduz o abl antigo)
#   python simclr_2x2_treino.py --aug crop_semcor --seed 0 --smoke   (teste rapido, CPU ok)
# ============================================================================
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint_sequential
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--aug", required=True,
                choices=["crop_semcor", "semcrop_comcor", "full_semgray", "full_semrot",
                         "full", "legacy_no_scale_color"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=20)          # orcamento por epocas (= receita completa)
ap.add_argument("--wallclock-h", type=float, default=11.5)  # rede de seguranca (job de 12 h)
ap.add_argument("--root", default=os.environ.get("WVC_ROOT", "."))
ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
ap.add_argument("--smoke", action="store_true", help="64 imagens, 2 'epocas' de 2 passos, CPU ok")
args = ap.parse_args()

ROOT = Path(args.root)
MM_PATH = ROOT / "nct100k_u8.mmap"
CKPT_DIR = ROOT / "camera_ready" / "ckpt_2x2"; CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = ROOT / "camera_ready" / "out" / "B2_1"; LOG_DIR.mkdir(parents=True, exist_ok=True)
TAG = f"s{args.seed}_{args.aug}" + ("_SMOKE" if args.smoke else "")

# ------------------- protocolo travado (revisao 1) -------------------
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
COUNTS_100K = [10407, 10566, 11512, 11557, 8896, 13536, 8763, 10446, 14317]
INTERNAL_SPLIT_SEED, INTERNAL_TEST_SIZE = 20260816, 0.20
BACKBONE, PROJ_DIM, TEMP = "resnet50", 128, 0.2
BATCH, EPOCHS_CAP, WARMUP_EPOCH = 256, 200, 10
BASE_LR, WEIGHT_DECAY, AMP = 1e-3, 1e-4, True
GRAD_CKPT = True
MONITOR_EVERY, MONITOR_PER_CLASS, KNN_K = 5, 40, 20
PLATEAU_DELTA, PLATEAU_PATIENCE, COLLAPSE_STD_MIN = 0.3, 4, 1e-3
NORM_MEAN, NORM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = False      # rev.1 nao ligava; manter
print("torch", torch.__version__, "| device", DEVICE,
      "|", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu", flush=True)

y_100k = np.repeat(np.arange(9), COUNTS_100K)
N = len(y_100k)
idx_train_pool, idx_eval_int = train_test_split(np.arange(N), test_size=INTERNAL_TEST_SIZE,
                                                stratify=y_100k, random_state=INTERNAL_SPLIT_SEED)
idx_train_pool = np.sort(idx_train_pool); y_train_pool = y_100k[idx_train_pool]
assert len(idx_train_pool) == 80000


def _carve_monitor(idx_pool, y_pool, per_class, seed=0):
    rng = np.random.default_rng(seed)
    tr, ev = [], []
    for ci in range(len(CLASSES)):
        pos = np.where(y_pool == ci)[0]
        rng.shuffle(pos)
        take = min(2 * per_class, len(pos)); half = take // 2
        tr += list(idx_pool[pos[:half]]); ev += list(idx_pool[pos[half:2 * half]])
    return np.array(sorted(tr)), np.array(sorted(ev))


mon_tr_idx, mon_ev_idx = _carve_monitor(idx_train_pool, y_train_pool, MONITOR_PER_CLASS)
idx_pretrain = idx_train_pool.copy()
if args.smoke:
    idx_pretrain = idx_pretrain[:64 * 4]; mon_tr_idx = mon_tr_idx[::10]; mon_ev_idx = mon_ev_idx[::10]


class RandomRot90:
    def __call__(self, img):
        k = int(torch.randint(0, 4, (1,)).item())
        return img.rotate(90 * k) if k else img


def _blur():
    return T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5)


def build_ssl_transform(aug):
    crop = T.RandomResizedCrop(224, scale=(0.2, 1.0))
    nocrop = T.Compose([T.Resize(224), T.CenterCrop(224)])
    jitter = T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
    gray = T.RandomGrayscale(p=0.2)
    flips = [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
    tail = [_blur(), T.ToTensor(), T.Normalize(NORM_MEAN, NORM_STD)]
    spatial = T.Compose(flips + [RandomRot90()])
    if aug == "full":                  # = rev.1 "full"
        steps = [crop, spatial, T.Compose([jitter, gray])]
    elif aug == "legacy_no_scale_color":   # = rev.1 "ablated_no_scale_color" (SimCLR-abl do paper)
        steps = [nocrop, spatial]
    elif aug == "crop_semcor":         # NOVA: com RandomResizedCrop, sem jitter e sem grayscale
        steps = [crop, spatial]
    elif aug == "semcrop_comcor":      # NOVA: Resize+CenterCrop, com jitter e grayscale
        steps = [nocrop, spatial, T.Compose([jitter, gray])]
    elif aug == "full_semgray":        # opcional: full sem RandomGrayscale
        steps = [crop, spatial, jitter]
    elif aug == "full_semrot":         # opcional: full sem rot90 (flips mantidos)
        steps = [crop, T.Compose(flips), T.Compose([jitter, gray])]
    else:
        raise ValueError(aug)
    return T.Compose(steps + tail)


eval_transform = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize(NORM_MEAN, NORM_STD)])


class Src:
    def __init__(self): self._mm = None
    def get(self, i):
        if self._mm is None:
            self._mm = np.memmap(MM_PATH, dtype=np.uint8, mode="r", shape=(N, 224, 224, 3))
        return Image.fromarray(np.asarray(self._mm[i]))


class TwoView(Dataset):
    def __init__(self, idx, t): self.idx, self.t, self.s = np.asarray(idx), t, Src()
    def __len__(self): return len(self.idx)
    def __getitem__(self, k):
        img = self.s.get(int(self.idx[k])); return self.t(img), self.t(img)


class EvalDS(Dataset):
    def __init__(self, idx): self.idx, self.s = np.asarray(idx), Src()
    def __len__(self): return len(self.idx)
    def __getitem__(self, k): return eval_transform(self.s.get(int(self.idx[k])))


class SimCLRNet(nn.Module):
    def __init__(self, backbone_name, proj_dim, grad_ckpt=True):
        super().__init__()
        base = torchvision.models.__dict__[backbone_name](weights=None)
        self.feat_dim = base.fc.in_features
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.stages = nn.Sequential(base.layer1, base.layer2, base.layer3, base.layer4)
        self.avgpool = base.avgpool
        self.grad_ckpt = grad_ckpt
        self.proj = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim), nn.ReLU(inplace=True),
                                  nn.Linear(self.feat_dim, proj_dim))

    def encode(self, x):
        x = self.stem(x)
        if self.grad_ckpt and self.training:
            x = checkpoint_sequential(self.stages, 4, x, use_reentrant=False)
        else:
            x = self.stages(x)
        return torch.flatten(self.avgpool(x), 1)

    def forward(self, x):
        h = self.encode(x)
        return h, F.normalize(self.proj(h), dim=1)


def nt_xent(z, temp):
    N2 = z.shape[0]; Nb = N2 // 2
    sim = (z @ z.t()) / temp
    sim.fill_diagonal_(float("-inf"))
    pos = (torch.arange(N2, device=z.device) + Nb) % N2
    return F.cross_entropy(sim, pos)


@torch.no_grad()
def encode(model, idx, batch=128):
    model.eval()
    dl = DataLoader(EvalDS(idx), batch_size=batch, shuffle=False, num_workers=args.workers, pin_memory=True)
    out = []
    for xb in dl:
        xb = xb.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
        out.append(model.encode(xb).float().cpu().numpy())
    return np.concatenate(out)


def knn_monitor(model):
    Xtr = encode(model, mon_tr_idx); ytr = y_100k[mon_tr_idx]
    Xev = encode(model, mon_ev_idx); yev = y_100k[mon_ev_idx]
    Ftr = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-9)
    Fev = Xev / (np.linalg.norm(Xev, axis=1, keepdims=True) + 1e-9)
    idx = np.argsort(-(Fev @ Ftr.T), axis=1)[:, :min(KNN_K, len(ytr))]
    pred = [np.unique(ytr[r], return_counts=True) for r in idx]
    pred = np.array([v[np.argmax(c)] for v, c in pred])
    return 100 * balanced_accuracy_score(yev, pred), float(Fev.std(axis=0).mean())


def _ftz(_):
    # flush-to-zero de denormais nos workers: o GaussianBlur(23) em PIL fica ate 10x mais rapido;
    # a saida e bit a bit identica (conferido em 60 pares imagem x sigma)
    torch.set_flush_denormal(True)


def cosine_warmup_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * p))


def main():
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = SimCLRNet(BACKBONE, PROJ_DIM, GRAD_CKPT).to(DEVICE).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(AMP and DEVICE == "cuda"))
    bs = 32 if args.smoke else BATCH
    dl = DataLoader(TwoView(idx_pretrain, build_ssl_transform(args.aug)), batch_size=bs, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True,
                    persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers > 0 else None,
                    worker_init_fn=_ftz if args.workers > 0 else None)
    steps_ep = len(dl)
    total_steps = EPOCHS_CAP * steps_ep              # agenda de 200 epocas truncada (rev.1)
    n_epochs = 2 if args.smoke else args.epochs
    hist, best_ba, best_ep, plateau, status = [], -1.0, -1, 0, "running"
    t0, gstep = time.time(), 0
    best_path = CKPT_DIR / f"simclr_{TAG}_best.pt"
    print(f"[{TAG}] aug={args.aug} steps/ep={steps_ep} epocas={n_epochs} batch={bs}", flush=True)
    for ep in range(n_epochs):
        model.train(); ep_loss, nb = 0.0, 0
        for v1, v2 in dl:
            lr = cosine_warmup_lr(gstep, total_steps, WARMUP_EPOCH * steps_ep, BASE_LR)
            for g in opt.param_groups: g["lr"] = lr
            x = torch.cat([v1, v2], 0).to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE, enabled=(AMP and DEVICE == "cuda")):
                _, z = model(x); loss = nt_xent(z, TEMP)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            ep_loss += loss.item(); nb += 1; gstep += 1
            if args.smoke and nb >= 2: break
        el = (time.time() - t0) / 3600
        print(f"  [{TAG}] fim da epoca {ep+1}: {el:.2f} h acumuladas ({el/(ep+1):.3f} h/epoca)", flush=True)
        mon_every = 1 if args.smoke else MONITOR_EVERY
        if (ep + 1) % mon_every == 0 or ep == n_epochs - 1:
            ba, std = knn_monitor(model)
            hist.append(dict(epoch=ep + 1, loss=ep_loss / max(nb, 1), knn_ba=ba, std=std, lr=lr, elapsed_h=el))
            sd = model.state_dict()
            meta = {"state_dict": sd, "epoch": ep + 1, "knn_ba": ba, "aug": args.aug, "seed": args.seed,
                    "backbone": BACKBONE, "batch": bs}
            torch.save(meta, CKPT_DIR / f"simclr_{TAG}_ep{ep+1}.pt")      # todos os monitorados
            improved = ba > best_ba + 1e-9
            if improved:
                best_ba, best_ep = ba, ep + 1
                torch.save(meta, best_path)
            plateau = plateau + 1 if (ba - best_ba) < PLATEAU_DELTA - 1e-9 and not improved else 0
            print(f"  [{TAG}] ep {ep+1:3d} loss {ep_loss/max(nb,1):.3f} knn_ba {ba:5.2f} std {std:.4f} "
                  f"lr {lr:.2e} {el:.2f}h (best {best_ba:.2f}@{best_ep})", flush=True)
            if std < COLLAPSE_STD_MIN: status = "colapso"; break
            if plateau >= PLATEAU_PATIENCE: status = "plato"; break
        if el >= args.wallclock_h: status = "timebox"; break
    else:
        status = "epocas_orcamento"
    import pandas as pd
    pd.DataFrame(hist).to_csv(LOG_DIR / f"convergencia_{TAG}.csv", index=False)
    summ = dict(tag=TAG, aug=args.aug, seed=args.seed, status=status, best_knn_ba=round(best_ba, 2),
                best_epoch=best_ep, epochs_run=hist[-1]["epoch"] if hist else 0,
                final_loss=hist[-1]["loss"] if hist else None, wallclock_h=round((time.time() - t0) / 3600, 2),
                gpu=torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
                best_ckpt=str(best_path), torch=torch.__version__)
    (LOG_DIR / f"convergencia_{TAG}.json").write_text(json.dumps(summ, indent=2))
    print("FIM", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
