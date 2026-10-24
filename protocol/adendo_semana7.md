# Adendo ao registro final - Semana 7 (reanalise sem refit)
**Gerado em:** 2026-08-18 09:26
**Pre-commit do 7k:** `df4b2937407f9b87` (inalterado; nenhuma predicao nova foi gerada).

## Convencao estatistica adotada
Duas fontes de incerteza, reportadas lado a lado e nunca somadas: IC95% sobre 5 seeds de rotulo (subamostragem) e IC95% bootstrap estratificado sobre os 7.180 patches (B=2000). Em frac=100% so existe a segunda. A tabela antiga (`estatistica_pareada_7k.csv`) era de **seed 0** e fica SUPERSEDIDA por `estatistica_pareada_7k_v2.csv`.

Vies maximo da convencao antiga: **10.44 pt**. Nenhum veredito muda de sinal: True.

## Estatistica pareada (v2)
| familia             | A            | B            | cond   |   frac |   dif |   ic_seed |   boot_lo |   boot_hi |   p_holm | sig   |
|:--------------------|:-------------|:-------------|:-------|-------:|------:|----------:|----------:|----------:|---------:|:------|
| confirmatoria       | SimCLR_s0    | LBP-COR-H    | RAW    |   0.01 |  7.03 |      4.03 |      6.18 |      7.83 |        0 | True  |
| confirmatoria       | Phikon       | LBP-COR-H    | RAW    |   0.01 | 15.37 |      3.58 |     14.5  |     16.2  |        0 | True  |
| confirmatoria       | SimCLR_s0    | LBP-COR-H    | PCA32  |   0.01 | 12.12 |      5.65 |     11.16 |     13.06 |        0 | True  |
| confirmatoria       | Phikon       | LBP-COR-H    | PCA32  |   0.01 | 20.48 |      3.98 |     19.58 |     21.4  |        0 | True  |
| confirmatoria       | Phikon       | SimCLR_s0    | RAW    |   0.01 |  8.34 |      2.37 |      7.69 |      9    |        0 | True  |
| confirmatoria       | LBP-COR-H    | LBP          | RAW    |   0.01 |  5.55 |      3.64 |      4.67 |      6.46 |        0 | True  |
| confirmatoria       | SimCLR_s0    | LBP-COR-H    | RAW    |   0.05 |  5.57 |      1.86 |      4.73 |      6.43 |        0 | True  |
| confirmatoria       | Phikon       | LBP-COR-H    | RAW    |   0.05 | 11.48 |      2.4  |     10.57 |     12.39 |        0 | True  |
| confirmatoria       | SimCLR_s0    | LBP-COR-H    | PCA32  |   0.05 | 10.83 |      1.8  |      9.79 |     11.86 |        0 | True  |
| confirmatoria       | Phikon       | LBP-COR-H    | PCA32  |   0.05 | 18.13 |      2.04 |     17.17 |     19.11 |        0 | True  |
| confirmatoria       | Phikon       | SimCLR_s0    | RAW    |   0.05 |  5.91 |      0.82 |      5.24 |      6.56 |        0 | True  |
| confirmatoria       | LBP-COR-H    | LBP          | RAW    |   0.05 |  8.08 |      1.98 |      7.04 |      9.14 |        0 | True  |
| confirmatoria       | SimCLR_s0    | LBP-COR-H    | RAW    |   1    | 10.06 |    nan    |      8.94 |     11.21 |        0 | True  |
| confirmatoria       | Phikon       | LBP-COR-H    | RAW    |   1    | 13.49 |    nan    |     12.46 |     14.54 |        0 | True  |
| confirmatoria       | SimCLR_s0    | LBP-COR-H    | PCA32  |   1    | 13.29 |    nan    |     12.05 |     14.55 |        0 | True  |
| confirmatoria       | Phikon       | LBP-COR-H    | PCA32  |   1    | 17.08 |    nan    |     15.97 |     18.24 |        0 | True  |
| confirmatoria       | Phikon       | SimCLR_s0    | RAW    |   1    |  3.44 |    nan    |      2.57 |      4.34 |        0 | True  |
| confirmatoria       | LBP-COR-H    | LBP          | RAW    |   1    |  4.75 |    nan    |      3.56 |      5.95 |        0 | True  |
| exploratoria_poshoc | SimCLR_s0    | CLASSICO-MAX | RAW    |   0.01 |  7.07 |      1.44 |      6.3  |      7.84 |        0 | True  |
| exploratoria_poshoc | Phikon       | CLASSICO-MAX | RAW    |   0.01 | 15.41 |      1.99 |     14.69 |     16.11 |        0 | True  |
| exploratoria_poshoc | SimCLR_s0    | CLASSICO-MAX | PCA32  |   0.01 | 12.78 |      2.76 |     11.8  |     13.8  |        0 | True  |
| exploratoria_poshoc | Phikon       | CLASSICO-MAX | PCA32  |   0.01 | 21.14 |      2.05 |     20.19 |     22.1  |        0 | True  |
| exploratoria_poshoc | CLASSICO-MAX | LBP          | RAW    |   0.01 |  5.52 |      1.25 |      4.7  |      6.35 |        0 | True  |
| exploratoria_poshoc | SimCLR_s0    | CLASSICO-MAX | RAW    |   0.05 |  5.68 |      1.44 |      4.85 |      6.54 |        0 | True  |
| exploratoria_poshoc | Phikon       | CLASSICO-MAX | RAW    |   0.05 | 11.59 |      2.22 |     10.74 |     12.4  |        0 | True  |
| exploratoria_poshoc | SimCLR_s0    | CLASSICO-MAX | PCA32  |   0.05 | 11.1  |      1.02 |     10    |     12.23 |        0 | True  |
| exploratoria_poshoc | Phikon       | CLASSICO-MAX | PCA32  |   0.05 | 18.4  |      1.09 |     17.39 |     19.4  |        0 | True  |
| exploratoria_poshoc | CLASSICO-MAX | LBP          | RAW    |   0.05 |  7.96 |      1.62 |      6.95 |      8.97 |        0 | True  |
| exploratoria_poshoc | SimCLR_s0    | CLASSICO-MAX | RAW    |   1    |  3.59 |    nan    |      2.5  |      4.69 |        0 | True  |
| exploratoria_poshoc | Phikon       | CLASSICO-MAX | RAW    |   1    |  7.02 |    nan    |      6    |      7.99 |        0 | True  |
| exploratoria_poshoc | SimCLR_s0    | CLASSICO-MAX | PCA32  |   1    | 11.43 |    nan    |     10.23 |     12.67 |        0 | True  |
| exploratoria_poshoc | Phikon       | CLASSICO-MAX | PCA32  |   1    | 15.23 |    nan    |     14.06 |     16.33 |        0 | True  |
| exploratoria_poshoc | CLASSICO-MAX | LBP          | RAW    |   1    | 11.22 |    nan    |     10.01 |     12.43 |        0 | True  |

## Interno vs externo (a contribuicao)
| classico     |   d |   int100 |   ext100 |   queda |   gapPhi_int |   gapPhi_ext |   ampl_Phi |   gapSSL_int |   gapSSL_ext |   ampl_SSL |
|:-------------|----:|---------:|---------:|--------:|-------------:|-------------:|-----------:|-------------:|-------------:|-----------:|
| Fractal      |  38 |    67.19 |    62.16 |    5.03 |        32.61 |        30.67 |      -1.94 |        30.77 |        27.23 |      -3.54 |
| Fractal-BM   |  35 |    75.92 |    68.08 |    7.84 |        23.88 |        24.74 |       0.86 |        22.04 |        21.31 |      -0.73 |
| LBP          |  54 |    82.55 |    74.59 |    7.96 |        17.25 |        18.24 |       0.99 |        15.41 |        14.8  |      -0.61 |
| LBP-H        |  54 |    83.45 |    73.58 |    9.87 |        16.35 |        19.25 |       2.9  |        14.51 |        15.82 |       1.31 |
| LBP-HF-H     | 176 |    84.19 |    74.38 |    9.81 |        15.61 |        18.45 |       2.84 |        13.77 |        15.01 |       1.24 |
| LBP-COR-H    | 246 |    97    |    79.34 |   17.66 |         2.8  |        13.49 |      10.69 |         0.96 |        10.06 |       9.1  |
| LBP-ENR      | 422 |    97.16 |    81.07 |   16.09 |         2.64 |        11.76 |       9.12 |         0.8  |         8.32 |       7.52 |
| CLASSICO-MAX | 457 |    97.55 |    85.81 |   11.74 |         2.25 |         7.02 |       4.77 |         0.41 |         3.59 |       3.18 |

## Custo do PCA32 por braco (mecanismo de H2)
| arm           |   0.01 |   0.05 |   0.1 |   0.25 |   1.0 |    d |
|:--------------|-------:|-------:|------:|-------:|------:|-----:|
| Fractal-BM    |   2.52 |  -0.34 | -1.12 |  -1.49 | -1.86 |   35 |
| Fractal       |  -1.39 |  -1.66 | -1.84 |  -1.73 | -1.08 |   38 |
| LBP           |   3.03 |   1.13 |  1.23 |   1.32 |  0.93 |   54 |
| LBP-H         |   3.37 |   2.3  |  1.9  |   1.54 |  0.88 |   54 |
| LBP-HF-H      |   2.13 |  11.49 | 12.79 |  14.31 | 15.32 |  176 |
| LBP-COR-H     |   5.52 |   6.3  |  7.85 |   8.5  |  4.3  |  246 |
| LBP-ENR       |   5.75 |   6.73 |  5.67 |   6.98 |  3.75 |  422 |
| CLASSICO-MAX  |   6.14 |   6.45 |  7.15 |   9.26 |  8.91 |  457 |
| Phikon        |   0.41 |  -0.36 |  0.03 |  -0.05 |  0.7  |  768 |
| SimCLR_abl_s0 |  -3.29 |   2.91 |  6.32 |  10.17 | 13.72 | 2048 |
| SimCLR_abl_s1 |  -0.56 |   3.92 |  7.79 |  10.3  | 14.03 | 2048 |
| SimCLR_s0     |   0.43 |   1.03 |  1.16 |   0.87 |  1.06 | 2048 |
| SimCLR_s1     |   0.21 |   0.9  |  1.24 |   1.4  |  1.96 | 2048 |
| SimCLR_s2     |  -0.36 |   0.26 |  0.47 |   0.34 |  1.05 | 2048 |

## H2: veredito por comparador
| aprendido   | classico     |   d |   frac |   gap_raw |   gap_pca |   delta | veredito   | paridade_restaurada   |
|:------------|:-------------|----:|-------:|----------:|----------:|--------:|:-----------|:----------------------|
| Phikon      | Fractal      |  38 |   0.05 |     34.32 |     33.01 |   -1.31 | ENCOLHE    | False                 |
| Phikon      | Fractal      |  38 |   1    |     30.67 |     28.89 |   -1.78 | ENCOLHE    | False                 |
| Phikon      | Fractal-BM   |  35 |   0.05 |     25.38 |     25.39 |    0.01 | PLANO      | False                 |
| Phikon      | Fractal-BM   |  35 |   1    |     24.74 |     22.18 |   -2.57 | ENCOLHE    | False                 |
| Phikon      | LBP          |  54 |   0.05 |     19.55 |     21.04 |    1.48 | CRESCE     | False                 |
| Phikon      | LBP          |  54 |   1    |     18.24 |     18.46 |    0.22 | PLANO      | False                 |
| Phikon      | LBP-H        |  54 |   0.05 |     20.21 |     22.86 |    2.65 | CRESCE     | False                 |
| Phikon      | LBP-H        |  54 |   1    |     19.25 |     19.42 |    0.17 | PLANO      | False                 |
| Phikon      | LBP-HF-H     | 176 |   0.05 |     24.3  |     36.15 |   11.85 | CRESCE     | False                 |
| Phikon      | LBP-HF-H     | 176 |   1    |     18.45 |     33.07 |   14.62 | CRESCE     | False                 |
| Phikon      | LBP-COR-H    | 246 |   0.05 |     11.48 |     18.13 |    6.66 | CRESCE     | False                 |
| Phikon      | LBP-COR-H    | 246 |   1    |     13.49 |     17.08 |    3.59 | CRESCE     | False                 |
| Phikon      | LBP-ENR      | 422 |   0.05 |     11.66 |     18.74 |    7.08 | CRESCE     | False                 |
| Phikon      | LBP-ENR      | 422 |   1    |     11.76 |     14.8  |    3.04 | CRESCE     | False                 |
| Phikon      | CLASSICO-MAX | 457 |   0.05 |     11.59 |     18.4  |    6.81 | CRESCE     | False                 |
| Phikon      | CLASSICO-MAX | 457 |   1    |      7.02 |     15.23 |    8.2  | CRESCE     | False                 |
| SimCLR_s0   | Fractal      |  38 |   0.05 |     28.41 |     25.71 |   -2.7  | ENCOLHE    | False                 |
| SimCLR_s0   | Fractal      |  38 |   1    |     27.23 |     25.09 |   -2.14 | ENCOLHE    | False                 |
| SimCLR_s0   | Fractal-BM   |  35 |   0.05 |     19.47 |     18.09 |   -1.38 | ENCOLHE    | False                 |
| SimCLR_s0   | Fractal-BM   |  35 |   1    |     21.31 |     18.38 |   -2.93 | ENCOLHE    | False                 |
| SimCLR_s0   | LBP          |  54 |   0.05 |     13.64 |     13.74 |    0.09 | PLANO      | False                 |
| SimCLR_s0   | LBP          |  54 |   1    |     14.8  |     14.67 |   -0.14 | PLANO      | False                 |
| SimCLR_s0   | LBP-H        |  54 |   0.05 |     14.3  |     15.56 |    1.26 | CRESCE     | False                 |
| SimCLR_s0   | LBP-H        |  54 |   1    |     15.82 |     15.63 |   -0.18 | PLANO      | False                 |
| SimCLR_s0   | LBP-HF-H     | 176 |   0.05 |     18.39 |     28.86 |   10.46 | CRESCE     | False                 |
| SimCLR_s0   | LBP-HF-H     | 176 |   1    |     15.01 |     29.27 |   14.26 | CRESCE     | False                 |
| SimCLR_s0   | LBP-COR-H    | 246 |   0.05 |      5.57 |     10.83 |    5.27 | CRESCE     | False                 |
| SimCLR_s0   | LBP-COR-H    | 246 |   1    |     10.06 |     13.29 |    3.23 | CRESCE     | False                 |
| SimCLR_s0   | LBP-ENR      | 422 |   0.05 |      5.75 |     11.44 |    5.7  | CRESCE     | False                 |
| SimCLR_s0   | LBP-ENR      | 422 |   1    |      8.32 |     11    |    2.68 | CRESCE     | False                 |
| SimCLR_s0   | CLASSICO-MAX | 457 |   0.05 |      5.68 |     11.1  |    5.42 | CRESCE     | False                 |
| SimCLR_s0   | CLASSICO-MAX | 457 |   1    |      3.59 |     11.43 |    7.85 | CRESCE     | False                 |

## H4: teste de grupo com IC e sensibilidade ao comparador
| aprendido   | classico     |   morfo |   textural |   dif_morfo_menos_textural |   ic_lo |   ic_hi |   p_boot |   amplitude_textural |   amplitude_morfo | suporta_H4   | status         |
|:------------|:-------------|--------:|-----------:|---------------------------:|--------:|--------:|---------:|---------------------:|------------------:|:-------------|:---------------|
| Phikon      | LBP-COR-H    |    9.45 |       5.59 |                       3.86 |    1.73 |    6.12 |   0.0005 |                33.85 |              1.37 | True         | pre-registrado |
| Phikon      | LBP          |   24.59 |      25.46 |                      -0.88 |   -3.42 |    1.74 |   0.7335 |                40.61 |             11.66 | False        | pos-hoc        |
| Phikon      | CLASSICO-MAX |    6.64 |       3.04 |                       3.61 |    1.53 |    5.73 |   0      |                25.15 |              3.72 | True         | pos-hoc        |
| SimCLR_s0   | LBP-COR-H    |    5.61 |       2.05 |                       3.56 |    1.25 |    6.03 |   0.0015 |                42.18 |              0.14 | True         | pre-registrado |
| SimCLR_s0   | LBP          |   20.74 |      21.92 |                      -1.18 |   -3.97 |    1.53 |   0.7995 |                47.95 |             10.43 | False        | pos-hoc        |
| SimCLR_s0   | CLASSICO-MAX |    2.79 |      -0.51 |                       3.3  |    1.08 |    5.64 |   0.0015 |                36.92 |              4.94 | True         | pos-hoc        |

## Gap por classe com IC (vs manchete)
|      |    gap |   ic_lo |   ic_hi | tipo_H4           | sig   |   gap_SSL |   ic_lo_SSL |   ic_hi_SSL |
|:-----|-------:|--------:|--------:|:------------------|:------|----------:|------------:|------------:|
| ADI  | -13.9  |  -16.07 |  -11.58 | textural          | True  |      2.54 |        0.97 |        4.19 |
| BACK |  53.84 |   50.65 |   57.38 | nao-preregistrada | True  |     53.84 |       50.65 |       57.38 |
| DEB  |  25.66 |   20.94 |   30.38 | nao-preregistrada | True  |     21.83 |       16.52 |       27.43 |
| LYM  |   0.63 |   -0.32 |    1.58 | nao-preregistrada | False |     -4.57 |       -6.62 |       -2.68 |
| MUC  |  12.95 |   10.82 |   15.07 | textural          | True  |      8.41 |        5.8  |       10.92 |
| MUS  |   3.38 |   -0.51 |    7.27 | textural          | False |    -22.47 |      -26.69 |      -18.24 |
| NORM |   8.77 |    6.61 |   10.93 | morfologica       | True  |      5.53 |        3.24 |        7.96 |
| STR  |  19.95 |   14.96 |   24.7  | textural          | True  |     19.71 |       14.01 |       24.94 |
| TUM  |  10.14 |    8.43 |   12    | morfologica       | True  |      5.68 |        3.57 |        7.7  |

## Nao-monotonicidade (10% menos 100%)
| arm          | cond   |   de |   para |   dif |   ic_lo |   ic_hi | sig   |
|:-------------|:-------|-----:|-------:|------:|--------:|--------:|:------|
| LBP-COR-H    | RAW    |  0.1 |      1 |  4.66 |    4.06 |    5.31 | True  |
| LBP-ENR      | RAW    |  0.1 |      1 |  1.11 |    0.42 |    1.82 | True  |
| CLASSICO-MAX | RAW    |  0.1 |      1 | -2.16 |   -2.82 |   -1.49 | True  |
| LBP          | RAW    |  0.1 |      1 | -0.13 |   -0.59 |    0.32 | False |
| Phikon       | RAW    |  0.1 |      1 |  0.52 |    0.08 |    0.96 | True  |
| SimCLR_s0    | RAW    |  0.1 |      1 | -0.51 |   -1.02 |    0.03 | False |

## Guardas que este notebook NAO remove
- H3-a continua sendo colapso de pretexto, nao ablacao de invariancia.
- SimCLR continua subtreinado: o gap FM-SSL e limite superior.
- Confirmacao de Macenko continua documental, nao empirica.
- Comparacoes contra CLASSICO-MAX e a releitura homogeneo-vs-irregular sao POS-HOC e tem de ser rotuladas assim.
