#!/bin/bash
# =============================================================================
# CPU (sem GPU de proposito: o monitor de GPU ociosa cancela jobs de GPU parados).
#   ETAPA=a  (independente, pode ir ja):
#            B1.1 extracao classica | B1.2 extracao classica | B1.3 proxies |
#            B1.4 varredura de PCA | refit interno + vazamento (B0.4/B1.5)
#   ETAPA=b  (depois de cr_gpu e da etapa a): avaliacao de B1.1 e B1.2 com os bracos aprendidos
#   ETAPA=c  (depois do array cr_2x2): probes das celulas novas do 2x2 (B2.1)
#   ETAPA=b0 (opcional) re-analise B0 (so le preds_7k/; ja rodada localmente, resultado identico)
# Submeter via camera_ready/hpc/submete_camera_ready.sh (cuida das dependencias).
# =============================================================================
#SBATCH --job-name=cr_cpu
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matheus25022@ilum.cnpem.br
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ilumpy
export CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg PYTHONUNBUFFERED=1
export WVC_ROOT="$SLURM_SUBMIT_DIR" CRC_VAL_DIR="${CRC_VAL_DIR:-CRC-VAL-HE-7K}" NCT_DIR="${NCT_DIR:-NCT-CRC-HE-100K}"
NCPU=${SLURM_CPUS_PER_TASK:-16}
S=camera_ready/scripts
ETAPA=${ETAPA:-a}
python -c "import sklearn, numpy, scipy, skimage; print('sklearn', sklearn.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'skimage', skimage.__version__)"

# pre-commit original nao pode mudar
HASH_ANTES=$(python -c "import json;print(json.load(open('out_semana56/precommit_7k.json'))['spec_hash'])")

case "$ETAPA" in
  a)
    # se camera_ready/out/ veio da maquina local (prévia, sklearn 1.6.1), descartar o que
    # tem de ser refeito no ambiente original - os scripts pulam o que ja existe
    rm -f camera_ready/out/interno/refit_interno_100.csv camera_ready/out/interno/*__interno.npy \
          camera_ready/out/preds_posaceite/B14__*
    # B0 primeiro: b1_interno_refit le camera_ready/out/B0/B0_4_semBACK_externo.csv
    OMP_NUM_THREADS=$NCPU python $S/b0_reanalise.py
    # refit interno cedo: a etapa c le a tabela interna que ele grava
    OMP_NUM_THREADS=4 N_PROC=4 python $S/b1_interno_refit.py
    # extracao: 1 thread de BLAS por processo
    OMP_NUM_THREADS=1 N_WORKERS=$NCPU python $S/b1_1_cor_interno.py extrai
    OMP_NUM_THREADS=1 python $S/b1_2_renorm_7k.py extrai
    OMP_NUM_THREADS=1 python $S/b1_3_proxies_normalizacao.py
    OMP_NUM_THREADS=4 N_PROC=4 python $S/b1_4_pca_sweep.py
    ;;
  b)
    OMP_NUM_THREADS=$NCPU python $S/b1_1_cor_interno.py avalia
    OMP_NUM_THREADS=$NCPU python $S/b1_2_renorm_7k.py avalia
    ;;
  c)
    OMP_NUM_THREADS=2 N_PROC=8 python $S/b2_1_probes_2x2.py ${TAGS:-s0_crop_semcor s0_semcrop_comcor}
    ;;
  b0)
    OMP_NUM_THREADS=$NCPU python $S/b0_reanalise.py
    ;;
esac

HASH_DEPOIS=$(python -c "import json;print(json.load(open('out_semana56/precommit_7k.json'))['spec_hash'])")
[[ "$HASH_ANTES" == "$HASH_DEPOIS" ]] || { echo "ERRO: precommit_7k.json mudou"; exit 1; }
N=$(ls out_semana56/preds_7k/*.npy | wc -l); [[ "$N" == 588 ]] || { echo "ERRO: preds_7k mudou ($N)"; exit 1; }
echo "OK etapa $ETAPA (pre-commit e preds_7k intactos)"
