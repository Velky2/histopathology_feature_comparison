# Registro - Semanas 5-6 (WVC 2026, §11): fatorial-nucleo, H3-b, H4 e TOQUE UNICO no 7k

**Data do registro:** 18/08/2026

**Ramo tomado:** fechamento experimental. Semanas 3-4 nao re-rodadas -> tres decisoes fechadas para o lado conservador (H3-a rebaixada; SimCLR declarado budget-limited/nao-convergido; manchete SSL treinada em 80k).

**Split desta execucao:** train_pool=80000 / eval_int=20000 (INTERNAL_SPLIT_SEED=20260816).

**Pre-commit do 7k:** hash `df4b2937407f9b87`, gravado em `precommit_7k.json`.

## 1. Numeros do paper (7k, coorte externa, toque unico)

| arm           |   0.01 |   0.05 |   0.1 |   0.25 |   1.0 |
|:--------------|-------:|-------:|------:|-------:|------:|
| CLASSICO-MAX  |  78.35 |  82.38 | 83.65 |  85.29 | 85.81 |
| Fractal       |  55.87 |  59.65 | 60.73 |  61.3  | 62.16 |
| Fractal-BM    |  67.28 |  68.58 | 68.53 |  68.14 | 68.08 |
| LBP           |  72.83 |  74.41 | 74.46 |  74.8  | 74.59 |
| LBP-COR-H     |  78.39 |  82.49 | 84    |  83.73 | 79.34 |
| LBP-ENR       |  77.16 |  82.31 | 82.18 |  83.62 | 81.07 |
| LBP-H         |  71.95 |  73.76 | 73.81 |  74.13 | 73.58 |
| LBP-HF-H      |  56.16 |  69.66 | 71.89 |  73.37 | 74.38 |
| Phikon        |  93.76 |  93.97 | 93.35 |  92.91 | 92.83 |
| SimCLR_abl_s0 |  54.34 |  63.48 | 65.48 |  69    | 70.63 |
| SimCLR_abl_s1 |  50.45 |  59.18 | 62.86 |  66.04 | 69.87 |
| SimCLR_s0     |  85.42 |  88.06 | 88.88 |  88.97 | 89.39 |
| SimCLR_s1     |  85.06 |  87.39 | 88.05 |  88.12 | 88.9  |
| SimCLR_s2     |  85.4  |  87.85 | 88.67 |  88.7  | 89.61 |

Condicao PCA32:

| arm           |   0.01 |   0.05 |   0.1 |   0.25 |   1.0 |
|:--------------|-------:|-------:|------:|-------:|------:|
| CLASSICO-MAX  |  72.21 |  75.92 | 76.5  |  76.03 | 76.9  |
| Fractal       |  57.26 |  61.31 | 62.57 |  63.04 | 63.24 |
| Fractal-BM    |  64.76 |  68.93 | 69.65 |  69.62 | 69.95 |
| LBP           |  69.8  |  73.29 | 73.23 |  73.48 | 73.66 |
| LBP-COR-H     |  72.87 |  76.19 | 76.15 |  75.22 | 75.04 |
| LBP-ENR       |  71.41 |  75.58 | 76.52 |  76.63 | 77.33 |
| LBP-H         |  68.58 |  71.46 | 71.91 |  72.59 | 72.7  |
| LBP-HF-H      |  54.03 |  58.17 | 59.1  |  59.06 | 59.06 |
| Phikon        |  93.35 |  94.32 | 93.32 |  92.96 | 92.12 |
| SimCLR_abl_s0 |  57.63 |  60.57 | 59.16 |  58.84 | 56.91 |
| SimCLR_abl_s1 |  51.02 |  55.26 | 55.07 |  55.74 | 55.84 |
| SimCLR_s0     |  84.99 |  87.02 | 87.72 |  88.09 | 88.33 |
| SimCLR_s1     |  84.85 |  86.49 | 86.81 |  86.72 | 86.94 |
| SimCLR_s2     |  85.76 |  87.59 | 88.2  |  88.36 | 88.56 |

## 2. Estatistica pareada (IC95% bootstrap sobre o teste)

| A         | B         | cond   |   frac |   diff |   ic_lo |   ic_hi |   p_boot_unilateral |   p_holm | significativo   |
|:----------|:----------|:-------|-------:|-------:|--------:|--------:|--------------------:|---------:|:----------------|
| SimCLR_s0 | LBP-COR-H | RAW    |   0.01 |   4.13 |    2.93 |    5.33 |              0      |   0      | True            |
| Phikon    | SimCLR_s0 | RAW    |   0.01 |  12.4  |   11.45 |   13.37 |              0      |   0      | True            |
| Phikon    | LBP-COR-H | RAW    |   0.01 |  16.53 |   15.39 |   17.68 |              0      |   0      | True            |
| LBP-COR-H | LBP       | RAW    |   0.01 |   5.63 |    4.43 |    6.85 |              0      |   0      | True            |
| SimCLR_s0 | LBP-COR-H | PCA32  |   0.01 |   1.68 |    0.51 |    2.93 |              0.0055 |   0.0055 | True            |
| Phikon    | LBP-COR-H | PCA32  |   0.01 |  13.81 |   12.72 |   14.92 |              0      |   0      | True            |
| SimCLR_s0 | LBP-COR-H | RAW    |   0.05 |   6.31 |    5.16 |    7.46 |              0      |   0      | True            |
| Phikon    | SimCLR_s0 | RAW    |   0.05 |   7.38 |    6.59 |    8.23 |              0      |   0      | True            |
| Phikon    | LBP-COR-H | RAW    |   0.05 |  13.69 |   12.6  |   14.83 |              0      |   0      | True            |
| LBP-COR-H | LBP       | RAW    |   0.05 |   6.98 |    5.78 |    8.25 |              0      |   0      | True            |
| SimCLR_s0 | LBP-COR-H | PCA32  |   0.05 |  13.65 |   12.39 |   14.93 |              0      |   0      | True            |
| Phikon    | LBP-COR-H | PCA32  |   0.05 |  21.29 |   20.1  |   22.49 |              0      |   0      | True            |
| SimCLR_s0 | LBP-COR-H | RAW    |   1    |  10.06 |    8.94 |   11.21 |              0      |   0      | True            |
| Phikon    | SimCLR_s0 | RAW    |   1    |   3.44 |    2.57 |    4.34 |              0      |   0      | True            |
| Phikon    | LBP-COR-H | RAW    |   1    |  13.49 |   12.46 |   14.54 |              0      |   0      | True            |
| LBP-COR-H | LBP       | RAW    |   1    |   4.75 |    3.56 |    5.95 |              0      |   0      | True            |
| SimCLR_s0 | LBP-COR-H | PCA32  |   1    |  13.29 |   12.05 |   14.55 |              0      |   0      | True            |
| Phikon    | LBP-COR-H | PCA32  |   1    |  17.08 |   15.97 |   18.24 |              0      |   0      | True            |

> **Desvio registrado do §7:** o Wilcoxon pareado sobre 5 seeds nao consegue p<0,05 (minimo bilateral = 0,0625). Substituido por diferenca pareada + IC bootstrap estratificado sobre os 7.180 patches + Holm. Valido tambem em frac=100%, onde nao ha seeds.

## 3. H2 no 7k (pendencia que bloqueava o paper desde a Semana 2)

| A         | B         |   frac |   raw |   pca |   delta | veredito   |
|:----------|:----------|-------:|------:|------:|--------:|:-----------|
| Phikon    | LBP-COR-H |   0.05 | 11.48 | 18.13 |    6.66 | gap CRESCE |
| Phikon    | LBP-COR-H |   1    | 13.49 | 17.08 |    3.59 | gap CRESCE |
| SimCLR_s0 | LBP-COR-H |   0.05 |  5.57 | 10.83 |    5.27 | gap CRESCE |
| SimCLR_s0 | LBP-COR-H |   1    | 10.06 | 13.29 |    3.23 | gap CRESCE |

## 4. H3-b: o que o enriquecimento classico comprou

|                       |   0.01 |   0.05 |   1.0 |
|:----------------------|-------:|-------:|------:|
| ('CLASSICO-MAX', 457) |  82.94 |  92.38 | 97.55 |
| ('Fractal', 38)       |  58.42 |  64.03 | 67.19 |
| ('Fractal-BM', 35)    |  70.56 |  73.26 | 75.92 |
| ('LBP', 54)           |  76.7  |  80.47 | 82.55 |
| ('LBP-COR-H', 246)    |  91.52 |  95.08 | 97    |
| ('LBP-ENR', 422)      |  81.01 |  91.83 | 97.16 |
| ('LBP-H', 54)         |  76.71 |  80.7  | 83.45 |
| ('LBP-HF-H', 176)     |  61.3  |  77.36 | 84.19 |

**Classico-manchete TRAVADO (antes do 7k):** `LBP-COR-H`. Regra pre-declarada: maior media entre as fracoes; empate dentro de 0,5 pt resolvido pela menor dimensao.

> **Guarda de leitura:** se o gap encolheu, isso significa 'parte do gap era pobreza do descritor', NAO 'o classico alcanca o aprendido'. O paradigma nao foi testado; o descritor foi.

## 5. H4 (decomposicao por classe)

|      |   Phikon - LBP-COR-H |   SimCLR_s0 - LBP-COR-H | tipo_H4           |
|:-----|---------------------:|------------------------:|:------------------|
| ADI  |               -13.9  |                    2.54 | textural          |
| BACK |                53.84 |                   53.84 | nao-preregistrada |
| DEB  |                25.66 |                   21.83 | nao-preregistrada |
| LYM  |                 0.63 |                   -4.57 | nao-preregistrada |
| MUC  |                12.95 |                    8.41 | textural          |
| MUS  |                 3.38 |                  -22.47 | textural          |
| NORM |                 8.77 |                    5.53 | morfologica       |
| STR  |                19.95 |                   19.71 | textural          |
| TUM  |                10.14 |                    5.68 | morfologica       |

> Pre-registro cobre 6 das 9 classes. BACK, DEB e LYM nao foram atribuidas a priori e ficam FORA do teste direcional.

## 6. Diagnostico de stain

- OD medio 100k: [0.3268, 0.7332, 0.3725] | 7k: [0.3707, 0.772, 0.3949]
- |delta| razao cromatica: 0.0201 | dispersao relativa: 1.212x
- Documental: o registro Zenodo 10.5281/zenodo.1214456 descreve NCT-CRC-HE-100K como normalizado por Macenko e distribui NCT-CRC-HE-100K-NONORM separadamente. Se NCT_DIR veio do primeiro, a condicao do nucleo (§3) bate com o declarado.

## 7. Guardas que sobrevivem para a escrita

- **(a)** Nenhum numero interno em tabela de desempenho. So o 7k.
- **(b)** H3-a e colapso de pretexto, nao ablacao de invariancia. Achado direcional + limitacao declarada. Nao sustenta a contribuicao central.
- **(c)** SimCLR subtreinado (parou por relogio na subida). O gap Phikon-SimCLR e LIMITE SUPERIOR. Proibido escrever 'SSL estaciona abaixo do FM'.
- **(d)** Reportar o PROBE, nunca o kNN-monitor, como qualidade do SimCLR.
- **(e)** H2: a afirmacao defensavel e a fraca - 'casar dimensao nao restaura paridade'. O gap CRESCE no eixo Phikon e fica PLANO no eixo SSL; o mecanismo de EVR difere.
- **(f)** Validacao assimetrica: so o Phikon tem ancora publicada (eva/kaiko, Tabela 3, 0,935).
- **(g)** H3-b move o lado classico, nao o aprendido.
- **(h)** Manchete SSL treinada em 80k (guarda transdutiva), nao 100k. Conservador; declarar.

## 8. Bibliografia - status

- **[FECHADO]** eva: kaiko.ai, I. Gatopoulos, N. Kanzig, R. Moser, S. Otalora, *eva: Evaluation framework for pathology foundation models*, MIDL 2024. Valor do Phikon-CRC na **Tabela 3** ('Linear probing evaluation of FMs, averaged balanced accuracy'): 0,935.
- **[FECHADO]** Dataset: Kather, Halama, Marx (2018), Zenodo 10.5281/zenodo.1214456.
- **[a conferir]** Ahonen et al. 2009 (LBP-HF); Kather et al. 2016 (Sci. Rep. 6:27988); Sarkar & Chaudhuri (DBC); Allain & Cloitre (lacunaridade); Backes & Bruno (Bouligand-Minkowski); Chen et al. 2020 (SimCLR); Filiot et al. (Phikon); Macenko et al. 2009 (ISBI, DOI 10.1109/ISBI.2009.5193250).

## 9. Itens abertos ao fim das Semanas 5-6

- Escrita (Semana 7): 6 paginas IEEE, titulo e resumo em ingles, checklist duplo-cego (§12).
- Trabalho futuro declarado: ablacao H3-a so-cor (mantendo crop); SimCLR convergido; pre-treino em 100k para a manchete.
- Conformidade: todo enquadramento interno fica FORA do PDF.
