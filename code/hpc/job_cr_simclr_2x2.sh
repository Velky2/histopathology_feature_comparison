#!/bin/bash
# =============================================================================
# B2.1 - celulas novas da ablacao 2x2 do SimCLR (1 celula por tarefa do array).
#   0 = crop_semcor     (com RandomResizedCrop, sem jitter/grayscale)   [obrigatoria]
#   1 = semcrop_comcor  (Resize+CenterCrop, com jitter/grayscale)        [obrigatoria]
#   2 = full_semgray    (opcional)      3 = full_semrot (opcional)
# Treina 20 epocas (receita da rev.1) e, sem liberar a GPU, extrai as features 100k+7k do
# checkpoint escolhido pela regra (melhor kNN-monitor) e, se for outro, do da epoca 20.
# Submeter:  sbatch --array=0-1 camera_ready/hpc/job_cr_simclr_2x2.sh
# Pico de memoria numa T4 ~15.4 GB de 16: se o cluster permitir escolher, peca a 4090
# (ex.: --gres=gpu:<tipo>:1 ou --constraint; ver  sinfo -o "%N %G").
# =============================================================================
#SBATCH --job-name=cr_2x2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matheus25022@ilum.cnpem.br
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"                  # pasta wvc/ (tem nct100k_u8.mmap, CRC-VAL-HE-7K/, camera_ready/)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ilumpy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WVC_ROOT="$SLURM_SUBMIT_DIR" CRC_VAL_DIR="${CRC_VAL_DIR:-CRC-VAL-HE-7K}"

CELULAS=(crop_semcor semcrop_comcor full_semgray full_semrot)
AUG=${CELULAS[$SLURM_ARRAY_TASK_ID]}
TAG="s0_${AUG}"
nvidia-smi --query-gpu=name,memory.total --format=csv
[[ -f nct100k_u8.mmap ]] || { echo "ERRO: nct100k_u8.mmap ausente (15 GB, gerado pelo wvc_s34_TREINO)"; exit 1; }

# reenvio limpo: nada de uma corrida anterior desta celula pode ser reaproveitado
rm -f camera_ready/ckpt_2x2/simclr_${TAG}_*.pt \
      camera_ready/cache_2x2/simclr_${TAG}_*.npy \
      camera_ready/out/preds_posaceite/B21__SimCLR_${TAG}__*.npy \
      camera_ready/out/B2_1/convergencia_${TAG}.*

python camera_ready/hpc/simclr_2x2_treino.py --aug "$AUG" --seed 0 --epochs 20 \
       --wallclock-h 17.5 --workers "$SLURM_CPUS_PER_TASK"

BEST=camera_ready/ckpt_2x2/simclr_${TAG}_best.pt
python camera_ready/hpc/extrai_gpu_camera_ready.py --modo ckpt --ckpt "$BEST" --tag "$TAG" --workers 8
EP20=camera_ready/ckpt_2x2/simclr_${TAG}_ep20.pt
BEST_EP=$(python -c "import json;print(json.load(open('camera_ready/out/B2_1/convergencia_${TAG}.json'))['best_epoch'])")
echo "checkpoint escolhido pela regra: epoca $BEST_EP"
if [[ -f "$EP20" && "$BEST_EP" != "20" ]]; then
  python camera_ready/hpc/extrai_gpu_camera_ready.py --modo ckpt --ckpt "$EP20" --tag "${TAG}_ep20" --workers 8
fi
echo "OK $TAG"; cat camera_ready/out/B2_1/convergencia_${TAG}.json
