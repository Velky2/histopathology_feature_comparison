#!/bin/bash
# =============================================================================
# B1.1 + B1.2 (bracos aprendidos): features do SimCLR s0 e do Phikon sobre
#   (a) a subamostra interna sob desvio de cor (4 condicoes x 4500 patches) e
#   (b) o 7k renormalizado para a OD do pool.
# Valida o pre-processamento contra os caches (cos > 0,999) antes de gravar.
# Custo: ~30-45 min numa T4. GPU ocupada o tempo todo (sem fase longa de CPU).
# =============================================================================
#SBATCH --job-name=cr_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matheus25022@ilum.cnpem.br
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ilumpy
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WVC_ROOT="$SLURM_SUBMIT_DIR" CRC_VAL_DIR="${CRC_VAL_DIR:-CRC-VAL-HE-7K}"
BRACOS="${BRACOS:-simclr_s0,phikon}"     # acrescente simclr_s1,simclr_s2 se sobrar tempo

nvidia-smi --query-gpu=name,memory.total --format=csv
python camera_ready/hpc/extrai_gpu_camera_ready.py --modo validar     --bracos "$BRACOS"
python camera_ready/hpc/extrai_gpu_camera_ready.py --modo cor_interno --bracos "$BRACOS"
python camera_ready/hpc/extrai_gpu_camera_ready.py --modo renorm_7k   --bracos "$BRACOS"
ls -la camera_ready/out/B1_1 camera_ready/out/B1_2
