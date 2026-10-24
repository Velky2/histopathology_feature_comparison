# ============================================================================
# Le as saidas das analises pos-aceite, aplica as REGRAS PRE-FIXADAS do
# adendo_posaceite.md e imprime os paragrafos finais (PT, convencao do PDF: ponto
# decimal) prontos para colar. Rodar depois que os jobs terminarem:
#     python camera_ready/scripts/gera_textos_finais.py  > camera_ready/out/TEXTOS_FINAIS.md
# Cada bloco diz de qual arquivo saiu e qual ramo da regra foi aplicado.
# ============================================================================
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CR_OUT, OUT56

out = []
P = out.append
f2 = lambda x: f"{x:.2f}"


def secao(t):
    P(f"\n## {t}\n")


# ---------------------------------------------------------------- B1.4
fp = CR_OUT / "B1_4" / "B1_4_gaps.csv"
secao("B1.4 - varredura de dimensao (IV-G, 'Dimensao')")
if fp.exists():
    G = pd.read_csv(fp)
    M = pd.read_csv(CR_OUT / "B1_4" / "B1_4_medias.csv")
    nao_pca = G[~G.cond.isin(["RAW"])]
    paridade = bool((nao_pca.gap <= 0).any())
    mn = nao_pca.loc[nao_pca.gap.idxmin()]
    ds = sorted({c for c in nao_pca.cond if c.startswith("PCA")}, key=lambda c: int(c[3:]))
    cst = M[(M.frac == 1.0)].pivot_table(index="arm", columns="cond", values="custo")
    k = M[M.cond.str.startswith("EVR") & (M.frac == 1.0)].pivot_table(index="arm", columns="cond", values="k")
    def c(arm, cond):
        try: return f2(cst.loc[arm, cond])
        except KeyError: return "n/d"
    P(f"_Fonte: {fp.name}; regra: paridade restaurada se alguma celula tiver gap <= 0 -> {'SIM' if paridade else 'NAO'}._\n")
    if not paridade:
        P(f"> *Dimensão.* Repetimos a condição de dimensão igualada com d ∈ {{{', '.join(d[3:] for d in ds)}}} e com a PCA "
          f"retendo 90% e 95% da variância, em 5% e 100% dos rótulos. Nenhuma das {len(nao_pca)} células fecha a diferença: "
          f"a menor é {mn.aprendido.replace('_s0','')} − {mn.classico.replace('LBP-COR-H','LBP-COL').replace('CLASSICO-MAX','CLASSIC-MAX')} "
          f"em {mn.cond.replace('PCA','d = ').replace('EVR','EVR ')}{'%' if mn.cond.startswith('EVR') else ''} e {int(mn.frac*100)}% (+{f2(mn.gap)}). "
          f"Em 100%, o custo da redução fica entre {f2(cst.loc['SimCLR_s0'].drop('RAW').min())} e {f2(cst.loc['SimCLR_s0'].drop('RAW').max())} "
          f"ponto para o SimCLR, e entre {f2(cst.loc['CLASSICO-MAX'].drop('RAW').min())} (d = 128) e {f2(cst.loc['CLASSICO-MAX'].drop('RAW').max())} (d = 8) "
          f"para o CLASSIC-MAX. "
          f"Retendo 95% da variância, o Phikon precisa de {int(k.loc['Phikon','EVR95'])} componentes, o SimCLR de {int(k.loc['SimCLR_s0','EVR95'])} "
          f"e o CLASSIC-MAX de {int(k.loc['CLASSICO-MAX','EVR95'])}, e a vantagem persiste. A conclusão de H2 vale para toda a faixa testada.")
    else:
        cel = nao_pca[nao_pca.gap <= 0]
        P("> *Dimensão.* ATENÇÃO: há células com gap ≤ 0 — a frase 'nunca restaura' precisa ser restrita. Células:\n")
        P(cel.to_markdown(index=False))
else:
    P("_B1_4_gaps.csv ausente — usar a versão 'em d = 32' do texto._")

# ---------------------------------------------------------------- B1.1
fp = CR_OUT / "B1_1" / "B1_1_resultado.csv"
secao("B1.1 - desvio de cor sintetico no interno (IV-G, 'Intervencao de cor')")
if fp.exists():
    R = pd.read_csv(fp)
    q = R.pivot(index="arm", columns="cond", values="queda9")
    dif = q.loc["LBP-COR-H", "a100"] - q.loc["LBP", "a100"]
    aprend = [a for a in ["SimCLR_s0", "Phikon"] if a in q.index]
    q_apr = {a: q.loc[a, "a100"] for a in aprend}
    ramo = "basta" if (dif >= 4.85 and all(v <= 1 for v in q_apr.values()) and len(aprend) == 2) else \
           ("nao explica" if dif < 2.4 else "explica parte")
    bk = R[(R.arm == "LBP-COR-H")].set_index("cond")
    P(f"_Fonte: {fp.name}; queda diferencial LBP-COL − LBP com α=1: {f2(dif)} (referência externa 9.70); "
      f"aprendidos com α=1: {', '.join(f'{a} {f2(v)}' for a, v in q_apr.items()) or 'ainda sem features'}; ramo: **{ramo}**._\n")
    linha_apr = ("; os braços aprendidos variam " + " e ".join(f"{f2(-v)} ({a.replace('_s0','')})" for a, v in q_apr.items())
                 if q_apr else "")
    P(f"> *Intervenção de cor.* Aplicamos ao split interno (500 patches por classe) um desvio global de cor, independente do rótulo, "
      f"do tamanho medido entre as coortes (ΔOD = [0.044, 0.039, 0.022], α = 1). O LBP-COL perde {f2(q.loc['LBP-COR-H','a100'])} pontos "
      f"(recall de BACK {f2(bk.loc['orig','rec_BACK'])} → {f2(bk.loc['a100','rec_BACK'])}; BACK→ADI {f2(bk.loc['a100','BACK_para_ADI'])}%), "
      f"o LBP {f2(q.loc['LBP','a100'])} e o CLASSIC-MAX {f2(q.loc['CLASSICO-MAX','a100'])}{linha_apr}. "
      f"A queda diferencial LBP-COL − LBP é {f2(dif)} pontos, ou {100*dif/9.70:.0f}% dos 9.70 observados entre as coortes "
      + (f"({f2(q.loc['LBP-COR-H','ajcomp'] - q.loc['LBP','ajcomp'])} com o desvio corrigido pela composição de classes, post hoc); "
         if 'ajcomp' in q.columns else "; ")
      + f"com α = 2 ela chega a {f2(q.loc['LBP-COR-H','a200'] - q.loc['LBP','a200'])} e, somando a dispersão medida, a "
      f"{f2(q.loc['LBP-COR-H','a100_disp'] - q.loc['LBP','a100_disp'])}. ")
    if ramo == "basta":
        P("> Um desvio global de cor da magnitude medida basta para explicar a maior parte da queda dos descritores de cor.")
    elif ramo == "explica parte":
        P("> Um desvio global de cor do tamanho medido explica parte da queda e não reproduz o colapso de BACK; o restante é sinal de coorte que não se reduz à média de cor.")
    else:
        P("> Um desvio global de cor dessa magnitude não reproduz a queda, o que aponta para diferenças entre as coortes que não se "
          "reduzem à média de cor (por exemplo, de processamento nos patches de fundo).")
else:
    P("_B1_1_resultado.csv ausente._")

# ---------------------------------------------------------------- B1.2
fp = CR_OUT / "B1_2" / "B1_2_resultado.csv"
secao("B1.2 - 7k renormalizado (IV-G, 'Intervencao de cor', 2a parte)")
if fp.exists():
    R = pd.read_csv(fp)
    P_ = R.pivot(index="arm", columns="cond", values="ba9"); B_ = R.pivot(index="arm", columns="cond", values="rec_BACK")
    g = (P_["renorm"] - P_["original"]); gb = (B_["renorm"] - B_["original"])
    ok_rep = R[(R.cond == "original")].concord_publicado.min()
    estaveis = [a for a in ["LBP", "SimCLR_s0", "Phikon"] if a in g.index]
    ramo = "positivo" if (g["LBP-COR-H"] >= 3 and gb["LBP-COR-H"] >= 25 and all(abs(g[a]) <= 1 for a in estaveis)) else "negativo"
    P(f"_Fonte: {fp.name}; concordância do refit com preds_7k publicadas: {ok_rep:.4f} (tem de ser 1.0); ramo: **{ramo}**._\n")
    P(f"> Na direção inversa, casar as estatísticas globais de densidade óptica do 7K com as do pool, sem usar rótulos, muda o LBP-COL "
      f"em {g['LBP-COR-H']:+.2f} pontos (BACK {gb['LBP-COR-H']:+.2f}), o LBP em {g['LBP']:+.2f}, o CLASSIC-MAX em {g['CLASSICO-MAX']:+.2f}"
      + "".join(f", o {a.replace('_s0','')} em {g[a]:+.2f}" for a in ["SimCLR_s0", "Phikon"] if a in g.index) + ". "
      + ("As estatísticas globais de cor explicam boa parte da queda." if ramo == "positivo"
         else "A queda não se reduz à estatística global de cor."))
else:
    P("_B1_2_resultado.csv ausente (roda só no HPC)._")

# ---------------------------------------------------------------- B1.3
fp = CR_OUT / "B1_3" / "B1_3_proxies.csv"
secao("B1.3 - proxies de normalizacao (IV-F, trecho D)")
if fp.exists():
    G = pd.read_csv(fp); info = json.loads((CR_OUT / "B1_3" / "B1_3_info.json").read_text())
    def v(coorte_sub, classe, col):
        r = G[G.coorte.str.contains(coorte_sub) & (G.classe == classe)]
        return float(r[col].iloc[0]) if len(r) else float("nan")
    tem7 = G.coorte.str.contains("7K").any()
    P(f"_Fonte: {fp.name}. Compressão TIFF/nomes: ver B1_3_info.json['tiff']._\n")
    txt = (f"> Como proxy da estimabilidade da normalização, a fração de patches em que a estimativa de Macenko é degenerada "
           f"(< 1% dos pixels com OD ≥ 0.15) é {100*v('100K','TODAS','frac_degenerado'):.1f}% no 100K"
           + (f" e {100*v('7K','TODAS','frac_degenerado'):.1f}% no 7K" if tem7 else "")
           + f", concentrada em BACK ({100*v('100K','BACK','frac_degenerado'):.1f}%"
           + (f" e {100*v('7K','BACK','frac_degenerado'):.1f}%" if tem7 else "") + ")")
    if "angulo_H_100k_vs_7k" in info:
        txt += (f"; os vetores de hematoxilina e eosina medianos diferem em {info['angulo_H_100k_vs_7k']:.1f}° e "
                f"{info['angulo_E_100k_vs_7k']:.1f}° entre as coortes")
    txt += "."
    P(txt)
else:
    P("_B1_3_proxies.csv ausente._")

# ---------------------------------------------------------------- B2.1
fp = CR_OUT / "B2_1" / "B2_1_tabela_2x2.csv"
secao("B2.1 - 2x2 do SimCLR (IV-E)")
if fp.exists():
    T = pd.read_csv(fp).set_index("celula")
    conv = {}
    for t in ["s0_crop_semcor", "s0_semcrop_comcor"]:
        j = CR_OUT / "B2_1" / f"convergencia_{t}.json"
        if j.exists(): conv[t] = json.loads(j.read_text())
    cs = pd.read_csv(CR_OUT / "B2_1" / "convergencia_s0_crop_semcor.csv") if (CR_OUT / "B2_1" / "convergencia_s0_crop_semcor.csv").exists() else None
    def lin(t):
        if t not in T.index: return None
        r = T.loc[t]; c = conv.get(t, {})
        return r, c
    a = lin("s0_crop_semcor"); b = lin("s0_semcrop_comcor")
    comp = T.loc["completo (paper)"]
    P(f"_Fonte: {fp.name}._\n")
    if a and b:
        (ra, ca), (rb, cb) = a, b
        P(f"> Para separar os dois fatores, treinamos as duas células que faltavam de um 2×2 (crop × cor), com a mesma receita, o mesmo "
          f"orçamento de 20 épocas e uma semente: com crop e sem cor, {f2(ra.ext_100)} no externo ({f2(ra.int_100)} no interno, loss final "
          f"{ca.get('final_loss', float('nan')):.2f}); sem crop e com cor, {f2(rb.ext_100)} ({f2(rb.int_100)}, loss "
          f"{cb.get('final_loss', float('nan')):.2f}), contra 89.39 (completo) e 70.63 (sem ambos).")
        amp = ra.get("ampl_vs_Phikon", np.nan)
        amp_full = 1.60   # completo s0: (99.80-97.96) interno -> (92.83-89.39) externo
        P(f"\n_Leitura pela regra: crop-sem-cor interno {f2(ra.int_100)} vs completo 97.96; amplificação vs Phikon "
          f"{f2(amp)} vs {amp_full:.2f} do completo (diferença {f2(amp - amp_full)}); loss crop-sem-cor "
          f"{ca.get('final_loss', float('nan')):.2f}, sem-crop-com-cor {cb.get('final_loss', float('nan')):.2f}._")
        if (amp - amp_full) >= 3 and abs(ra.int_100 - 97.96) <= 1.5:
            P("> Remover só a cor preserva a qualidade interna e piora a transferência, o que é consistente com a invariância de cor proteger as features aprendidas da mudança de coorte. Uma semente por célula nova; leitura direcional.")
        elif cb.get("final_loss", 9) < 1.9 and abs(ra.ext_100 - 89.39) <= 2:
            P("> A queda do SimCLR-abl vem do crop, que gera a dificuldade do pretexto; a cor, sozinha, custa "
              f"{f2(89.39 - ra.ext_100)} ponto(s) no externo. H3-a, na versão que isola a invariância de cor, não se sustenta. Uma semente por célula nova; leitura direcional.")
        elif (97.96 - ra.int_100) > 1.5 and ca.get("final_loss", 0) > 2.0:
            P("> Remover a cor degrada a representação mesmo com o pretexto viável, o que é consistente com H3-a para a invariância de cor. Uma semente por célula nova; leitura direcional.")
        else:
            P("> (nenhum ramo da regra se aplica limpo: descrever os números e declarar 'leitura inconclusiva com uma semente')")
    else:
        P("_alguma célula do 2×2 falhou — ver B2_1_tabela_2x2.csv_")
else:
    P("_B2_1_tabela_2x2.csv ausente._")

print("# Textos finais gerados das análises pós-aceite\n" + "\n".join(out))
