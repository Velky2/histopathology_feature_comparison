# Errata do adendo pós-aceite (escrita em 2026-10-22, tarde)

O `adendo_posaceite.md` (SHA-256 `8936f37b…`, 09:23:46) não é editado; as correções ficam aqui.

1. **"Gravado antes de qualquer predição nova" é impreciso.** As re-análises B0 (sem predições novas) foram calculadas às 09:15–09:20, antes do adendo. O refit interno (`b1_interno_refit.py`) começou às 09:21:47 e gravou predições **internas** (não do 7k) de 4 braços clássicos antes de 09:23:46. Nenhuma predição nova do 7k existia antes do adendo. As predições novas do 7k (B1.4, às 10:00) e a intervenção de cor no interno (B1.1, às 10:00) são posteriores.
2. **A regra de B0.4 (amplificação ≥ +3 sem BACK) não é pré-fixada.** Ela veio da triagem (20/10), escrita conhecendo uma prévia (≈ +3.5), e o B0.4 foi calculado antes do adendo. B0.2, B0.3 e B0.4 são **descritivos**.
3. **Condição nova em B1.1, post hoc:** `ajcomp` (OD + ΔOD ajustado pela composição de classes = [0.0489, 0.0792, 0.0376]). Motivo: a diferença entre as médias de OD das coortes mistura cor com composição de classes (o 7k tem mais ADI e MUC e menos DEB, MUS e STR). A regra de leitura continua a de α = 1; `ajcomp` é sensibilidade.
4. **B1.1, braços aprendidos:** a linha de base passa a ser a recalculada (`a000`), não o cache, para que a queda não inclua o ruído de recálculo.
5. **Bootstrap hierárquico (B0.1/B0.3) é descritivo.** Com 5 sementes, p < 1/32 não é sustentável no nível de semente.
