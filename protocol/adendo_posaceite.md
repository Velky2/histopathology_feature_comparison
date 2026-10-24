# Adendo pós-aceite — análises exploratórias para o camera-ready (WVC 2026)

**Gravado em:** 2026-10-22, antes de qualquer predição nova (o hash deste arquivo vai em `adendo_posaceite.sha256`).
**Pré-commit original do 7k:** `df4b2937407f9b87` (17/08/2026 22:13). **Não muda.** A Tab. I e a família confirmatória ficam como estão.

Tudo abaixo é **pós-revisão / exploratório**. Entra no paper numa subseção curta "Análises pós-revisão" e no suplemento do repositório, nunca na família confirmatória. Cada análise tem a regra de leitura fixada aqui, antes de rodar.

## Sem predições novas (re-análise de `preds_7k/`)

| ID | Análise | Regra de leitura (fixa) |
|---|---|---|
| B0.1 | p da Tab. IV: bootstrap pareado estratificado B = 20 000 (mesma semente 20260818), p = (k+1)/(B+1), unilateral na direção pré-registrada; t pareado sobre 5 sementes (gl = 4); bootstrap hierárquico sementes × patches; Holm por família (30 confirmatórias, 25 pós-hoc) | Substitui "p < 0,001" pelo maior p_Holm observado. Se alguma comparação confirmatória tiver p_Holm ≥ 0,05 em alguma fonte, isso é declarado no texto. |
| B0.2 | ICC intra-paciente crítico (efeito de desenho, m = 7180/50 e 7180/25) | Reportado como sensibilidade; não muda vereditos. |
| B0.3 | Comparações por semente de pré-treino do SimCLR (s0, s1, s2) e bootstrap hierárquico | A conclusão "SimCLR > clássico" vale se valer para a pior semente. |
| B0.4 | Acurácia balanceada de 8 classes (sem BACK), interno e externo | Se a amplificação sem BACK for ≥ +3 pontos nos dois eixos aprendidos, a assimetria é "presente nas 8 classes de tecido, com cerca de metade concentrada em BACK"; se < +3, a contribuição principal é reformulada em torno de BACK. |
| B0.5 | Ranking do gap por classe (6 e 9 classes) × comparador × fração | Ajusta o "em todas as configurações". |

## Com predições novas

| ID | Análise | Toca o 7k? | Regra de leitura (fixa) |
|---|---|---|---|
| B1.1 | Desvio de cor sintético, global e independente do rótulo, aplicado a uma subamostra estratificada do split interno (500 patches/classe, semente 20261022). Em densidade óptica (OD = −ln(x/255), clip 1e-3): (a) OD + α·Δ, Δ = OD̄(7k) − OD̄(100K) = [0,0439; 0,0388; 0,0224], α ∈ {0,5; 1; 2}; (b) α = 1 mais dispersão entre imagens inflada pela razão medida de DP por canal (7k/100K = [1,341; 1,030; 1,266]) em torno da média da coorte. Probes de 100% já ajustados no pool de 80K (não reajustados para a cor). | **não** | Queda diferencial de referência no externo: 17,66 − 7,96 = 9,70 pontos (LBP-COL − LBP). Se α = 1 reproduzir ≥ 50% disso (≥ 4,85 pontos de queda diferencial) e os braços aprendidos caírem ≤ 1 ponto: "um desvio global de cor da magnitude medida basta para explicar a maior parte da queda dos descritores de cor". Se reproduzir < 25% (< 2,4 pontos): "um desvio global de cor não explica a queda; o sinal de coorte não se restringe à média de cor". Entre os dois: "explica parte". Registrar também se aparece BACK → ADI. |
| B1.2 | 7k renormalizado para as estatísticas globais de OD do pool de 80K (média e DP por canal em nível de pixel, sobre 1000 imagens sorteadas de cada coorte, **nunca por classe**). Probes de 100% existentes. | sim | Se LBP-COL ganhar ≥ 3 pontos e o recall de BACK subir ≥ 25 pontos, com LBP e braços aprendidos variando ≤ 1 ponto: "as estatísticas globais de cor explicam boa parte da queda". Caso contrário: "a queda não se reduz à estatística global de cor". |
| B1.3 | Proxies de normalização por coorte e classe: fração de patches com estimativa de Macenko degenerada (< 1% dos pixels com OD > 0,15 em todos os canais), ângulo dos vetores H/E estimados contra a mediana da coorte, fração de pixels saturados, índice de blocagem JPEG 8×8 | sim (pixels) | Descritivo; chamado de "estimabilidade/consistência", nunca "taxa de sucesso". |
| B1.4 | Varredura de dimensão do PCA: d ∈ {8, 16, 64, 128} (d ≤ dimensão original) e PCA retendo 90% e 95% da variância, frações 5% (5 sementes) e 100%, todos os braços | sim | "Igualar a dimensão nunca restaura a paridade" se mantém se nenhuma célula (braço aprendido, comparador, d, fração) tiver gap ≤ 0. Caso contrário, restringir a frase às células em que vale. |
| B1.5 | Proxy de vazamento por vizinho mais próximo (cosseno) nos espaços Phikon, SimCLR e LBP-COL; duplicatas exatas por hash de pixel | só features | Descritivo; "proximidade não prova mesma lâmina". |
| B2.1 | Ablação fatorial 2 × 2 do SimCLR (semente 0), mesma receita da revisão 1 (cronograma de 200 épocas truncado, warm-up de 10, AdamW 1e-3, NT-Xent τ = 0,2, batch 256), **20 épocas** (= o que o timebox de 12 h permitiu à receita completa), seleção de checkpoint pela mesma regra (melhor kNN-monitor entre as épocas 5, 10, 15, 20, conjunto de monitor dentro do pool). Células novas: (i) com crop, sem cor; (ii) sem crop, com cor. Opcionais: só sem grayscale; só sem rot90. | sim | Guardas de pretexto: se uma célula chegar a loss ≈ 1,6 com kNN-monitor caindo, é colapso de pretexto (não invariância). Leitura: se "com crop, sem cor" mantiver o desempenho interno e cair forte no externo (amplificação ≥ +3 pontos contra o SimCLR completo), a invariância de cor protege a transferência; se cair já no interno com loss saudável (≈ 2,3), é efeito de invariância sobre a qualidade; uma semente por célula nova — declarado. |

## O que não muda
- A Tab. I, a Tab. IV confirmatória e os vereditos de H1–H4 (só a redação e os p).
- O 7k continua sem uso para seleção de modelo, hiperparâmetro ou checkpoint.
