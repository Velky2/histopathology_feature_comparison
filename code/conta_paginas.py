# ============================================================================
# Estimativa de paginas da versao condensada (camera_ready_PAPER_6paginas.md),
# calibrada pelo PDF aceito:
#   - PDF aceito = 6 paginas cheias + 24 linhas de referencias na 7a (~0.375 pag)
#     -> ~816 linhas de coluna (2 colunas x ~64 linhas por pagina);
#   - texto corrido original (sem titulo, floats, legendas e referencias) = 24,340
#     caracteres, ~57 caracteres por linha de coluna -> ~427 linhas;
#   - o resto (titulo, floats, legendas, titulos de secao, referencias) = ~389 linhas.
# Mudancas de floats da versao condensada: sai a antiga Tab. II (~20 linhas), as
# Figs. 1-2 vao a ~85% da altura (~-8), entra a ref. [18] (~+2), legendas um pouco maiores (~+2).
#   python conta_paginas.py [arquivo.md]
# ============================================================================
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
md = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "camera_ready_PAPER_6paginas.md"
s = md.read_text(encoding="utf-8")

# blocos ```text ... ``` = texto corrido; separa as legendas e a ref. nova (contadas nos floats)
blocos = re.findall(r"```text\n(.*?)```", s, flags=re.S)
CPL, LPP = 57, 128                          # caracteres por linha de coluna; linhas por pagina
ORIG_PROSA, ORIG_TOTAL = 24340, 816
ORIG_RESTO = ORIG_TOTAL - ORIG_PROSA / CPL
DELTA_FLOATS = -20 - 8 + 2 + 2
prosa = [b for b in blocos if not b.startswith("[18]") and not b.startswith("Acurácia balanceada interna")
         and not b.startswith("Recall por classe") and not b.startswith("Agradecemos")]
n = sum(len(b) for b in prosa)
# fins de paragrafo/item desperdicam linha; o original tinha ~52 paragrafos/itens (ja embutidos
# na calibracao), entao so o EXCESSO de paragrafos custa +0.5 linha cada
quebras = sum(b.count("\n") + 1 for b in prosa)
linhas = n / CPL + 0.5 * max(quebras - 52, 0)
total = linhas + ORIG_RESTO + DELTA_FLOATS + 1      # +1: agradecimentos
print(f"texto corrido: {n:,} caracteres em {len(prosa)} blocos ({quebras} paragrafos/itens) ~ {linhas:.0f} linhas de coluna")
print(f"original: {ORIG_PROSA:,} caracteres ~ {ORIG_PROSA/CPL:.0f} linhas; resto (titulo/floats/refs) ~ {ORIG_RESTO:.0f} linhas")
print(f"estimativa total: {total:.0f} linhas = {total/LPP:.2f} paginas (limite 6.00 = {6*LPP} linhas)")
folga = 6 * LPP - total
print(("folga" if folga >= 0 else "EXCESSO") + f": {abs(folga):.0f} linhas de coluna (~{abs(folga)*CPL:,.0f} caracteres)")
