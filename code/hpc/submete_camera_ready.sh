#!/bin/bash
# =============================================================================
# Submete tudo do camera-ready com as dependencias certas. Rodar no no de login, de
# dentro da pasta wvc/ (a mesma de out_semana56/, ckpt_simclr/, nct100k_u8.mmap):
#     bash camera_ready/hpc/submete_camera_ready.sh              # 2 celulas do 2x2
#     CELULAS=0-3 bash camera_ready/hpc/submete_camera_ready.sh  # + opcionais (ou 0,1,2 etc.)
# Confira antes:  sinfo -s  (nomes das particoes gpu/cpu) e  ls CRC-VAL-HE-7K | head
# =============================================================================
set -euo pipefail
mkdir -p logs
CELULAS=${CELULAS:-0-1}
NOMES=(s0_crop_semcor s0_semcrop_comcor s0_full_semgray s0_full_semrot)
TAGS=""
for i in $(echo "$CELULAS" | tr ',' ' '); do
  if [[ $i == *-* ]]; then for j in $(seq ${i%-*} ${i#*-}); do TAGS+="${NOMES[$j]} "; done
  else TAGS+="${NOMES[$i]} "; fi
done
TAGS=${TAGS% }

J2X2=$(sbatch --parsable --array=$CELULAS camera_ready/hpc/job_cr_simclr_2x2.sh)
JGPU=$(sbatch --parsable camera_ready/hpc/job_cr_extrai_gpu.sh)
JCA=$(sbatch --parsable --export=ALL,ETAPA=a camera_ready/hpc/job_cr_cpu.sh)
# b: avaliacao de B1.1/B1.2 com os aprendidos; afterany para nao travar se um script da etapa a falhar
JCB=$(sbatch --parsable --export=ALL,ETAPA=b --dependency=afterany:${JGPU}:${JCA} camera_ready/hpc/job_cr_cpu.sh)
# c: probes do 2x2; afterany nos dois (a tabela interna da etapa a e opcional para o b2_1)
JCC=$(sbatch --parsable --export=ALL,ETAPA=c,TAGS="$TAGS" --dependency=afterany:${J2X2}:${JCA} camera_ready/hpc/job_cr_cpu.sh)
cat <<EOF
submetidos:
  2x2 (GPU, array $CELULAS)  : $J2X2   tags: $TAGS
                                 o log imprime h/epoca; 20 epocas ~4-11 h numa T4
  extracao GPU (B1.1/B1.2)   : $JGPU   ~45 min
  CPU etapa a                : $JCA    ~3-5 h
  CPU etapa b (apos GPU e a) : $JCB    ~30 min
  CPU etapa c (apos 2x2 e a) : $JCC    ~2 h
acompanhar: squeue -u \$USER ; tail -f logs/cr_*.out
EOF
