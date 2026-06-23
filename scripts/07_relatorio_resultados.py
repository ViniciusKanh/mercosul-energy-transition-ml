# -*- coding: utf-8 -*-

"""
Script 07 - Relatorio consolidado dos resultados.

Objetivo:
Consolidar os principais achados quantitativos em relatorios Markdown, incluindo
validacao sem vazamento, robustez de clusters, interpretacao de perfis e
limitacoes metodologicas.

Execucao:
python scripts/07_relatorio_resultados.py
"""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.config import PROCESSED_DIR, TABLES_DIR, REPORTS_DIR, PAISES_MERCOSUL_ASSOCIADOS  # noqa: E402
from utils.io_utils import garantir_diretorios  # noqa: E402


def ler_csv_opcional(caminho: Path) -> pd.DataFrame | None:
    """Le CSV se existir; caso contrario, retorna None."""
    if caminho.exists():
        return pd.read_csv(caminho)
    return None


def tabela_markdown(df: pd.DataFrame | None, n: int = 10) -> str:
    """Converte DataFrame pequeno em Markdown."""
    if df is None or df.empty:
        return "Arquivo nao encontrado ou tabela vazia."
    return df.head(n).to_markdown(index=False)


def melhor_linha(df: pd.DataFrame | None, colunas_ordem: list[str]) -> pd.Series | None:
    """Retorna melhor linha segundo colunas de ordenacao decrescente."""
    if df is None or df.empty:
        return None

    colunas_existentes = [c for c in colunas_ordem if c in df.columns]
    if not colunas_existentes:
        return df.iloc[0]

    return df.sort_values(colunas_existentes, ascending=[False] * len(colunas_existentes)).iloc[0]


def resumo_base() -> dict[str, object]:
    """Resume base processada e cobertura temporal."""
    caminho = PROCESSED_DIR / "02_dataset_atributos.csv"
    if not caminho.exists():
        return {
            "qtd_observacoes": "nao disponivel",
            "qtd_paises": "nao disponivel",
            "ano_minimo": "nao disponivel",
            "ano_maximo": "nao disponivel",
            "paises": PAISES_MERCOSUL_ASSOCIADOS,
        }

    df = pd.read_csv(caminho)
    return {
        "qtd_observacoes": len(df),
        "qtd_paises": df["country"].nunique() if "country" in df.columns else "nao disponivel",
        "ano_minimo": int(df["year"].min()) if "year" in df.columns else "nao disponivel",
        "ano_maximo": int(df["year"].max()) if "year" in df.columns else "nao disponivel",
        "paises": sorted(df["country"].dropna().unique().tolist()) if "country" in df.columns else PAISES_MERCOSUL_ASSOCIADOS,
    }


def formatar_melhor_cluster(metricas_cluster: pd.DataFrame | None) -> str:
    """Resume melhor resultado de clusterizacao."""
    if metricas_cluster is None or metricas_cluster.empty:
        return "Metricas de clusterizacao ainda nao disponiveis."

    ordenado = metricas_cluster.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True])
    melhor = ordenado.iloc[0]
    return (
        f"Melhor configuracao inicial: `{melhor.get('modelo_cluster')}` com k={melhor.get('k')}, "
        f"silhouette={melhor.get('silhouette'):.3f}, "
        f"Davies-Bouldin={melhor.get('davies_bouldin'):.3f} e "
        f"Calinski-Harabasz={melhor.get('calinski_harabasz'):.3f}."
    )


def formatar_melhor_classificacao(metricas: pd.DataFrame | None) -> str:
    """Resume melhor classificador."""
    melhor = melhor_linha(metricas, ["f1_macro", "balanced_accuracy"])
    if melhor is None:
        return "Metricas de classificacao ainda nao disponiveis."

    return (
        f"Melhor modelo: `{melhor.get('modelo')}` com accuracy={melhor.get('accuracy'):.3f}, "
        f"balanced accuracy={melhor.get('balanced_accuracy'):.3f} e F1 macro={melhor.get('f1_macro'):.3f}."
    )


def avaliar_viabilidade(
    metricas_classificacao: pd.DataFrame | None,
    metricas_cluster: pd.DataFrame | None,
    metricas_sem_vazamento: pd.DataFrame | None,
) -> str:
    """Aplica criterios simples de decisao experimental."""
    pareceres = []

    if metricas_cluster is not None and not metricas_cluster.empty:
        melhor_silhouette = metricas_cluster["silhouette"].max()
        if melhor_silhouette >= 0.30:
            pareceres.append("Clusterizacao com separacao inicial aceitavel para exploracao cientifica.")
        else:
            pareceres.append("Clusterizacao com separacao fraca; revisar variaveis e recorte temporal.")
    else:
        pareceres.append("Metricas de clusterizacao ainda nao disponiveis.")

    if metricas_classificacao is not None and not metricas_classificacao.empty:
        melhor_f1 = metricas_classificacao["f1_macro"].max()
        if melhor_f1 >= 0.70:
            pareceres.append("Classificacao supervisionada reproduz bem os perfis derivados da clusterizacao original.")
        else:
            pareceres.append("Classificacao supervisionada abaixo do criterio desejavel.")
    else:
        pareceres.append("Metricas de classificacao ainda nao disponiveis.")

    if metricas_sem_vazamento is not None and not metricas_sem_vazamento.empty:
        melhor_f1_sem = metricas_sem_vazamento["f1_macro"].max()
        if melhor_f1_sem >= 0.70:
            pareceres.append("Validacao sem vazamento manteve desempenho adequado em teste temporal.")
        else:
            pareceres.append("Validacao sem vazamento reduziu desempenho; discutir essa diferenca no artigo.")
    else:
        pareceres.append("Validacao sem vazamento ainda nao disponivel.")

    return "\n".join([f"- {p}" for p in pareceres])


def resumir_cluster_intermediario(cluster_intermediario: pd.DataFrame | None) -> str:
    """Gera nota metodologica sobre frequencia e cobertura do cluster intermediario."""
    if cluster_intermediario is None or cluster_intermediario.empty:
        return "A tabela detalhada do cluster intermediario ainda nao esta disponivel."

    qtd_registros = len(cluster_intermediario)
    qtd_paises = cluster_intermediario["country"].nunique() if "country" in cluster_intermediario.columns else 0
    paises = sorted(cluster_intermediario["country"].dropna().unique().tolist()) if "country" in cluster_intermediario.columns else []
    ano_min = int(cluster_intermediario["year"].min()) if "year" in cluster_intermediario.columns else "nao disponivel"
    ano_max = int(cluster_intermediario["year"].max()) if "year" in cluster_intermediario.columns else "nao disponivel"

    texto = (
        f"O cluster intermediario contem {qtd_registros} registros, cobre {qtd_paises} pais(es) "
        f"e aparece entre {ano_min} e {ano_max}."
    )

    if qtd_paises == 1:
        texto += (
            f" Como esta associado apenas a {paises[0]}, deve ser tratado com cautela: "
            "ele pode representar uma trajetoria nacional especifica, e nao um padrao regional amplo."
        )
    elif qtd_paises > 1:
        texto += (
            f" Os paises associados sao: {', '.join(paises)}. Ainda assim, sua frequencia deve ser "
            "comparada aos demais perfis antes de formular conclusoes fortes."
        )

    return texto


def resumir_diagnostico_matriz(diagnostico: pd.DataFrame | None) -> str:
    """Explica automaticamente se alguma classe nao apareceu no teste temporal."""
    if diagnostico is None or diagnostico.empty:
        return "Diagnostico da matriz de confusao ainda nao disponivel."

    linha_zerada = (
        diagnostico["linha_zerada"]
        if "linha_zerada" in diagnostico.columns
        else pd.Series(False, index=diagnostico.index)
    )
    coluna_zerada = (
        diagnostico["coluna_zerada"]
        if "coluna_zerada" in diagnostico.columns
        else pd.Series(False, index=diagnostico.index)
    )

    ausentes = diagnostico[
        diagnostico["presente_no_teste"].astype(str).str.lower().isin(["false", "0"])
        | linha_zerada.astype(str).str.lower().isin(["true", "1"])
        | coluna_zerada.astype(str).str.lower().isin(["true", "1"])
    ]
    if ausentes.empty:
        return "Todas as classes consideradas na matriz de confusao aparecem no conjunto de teste temporal."

    classes = ", ".join(ausentes["classe"].astype(str).tolist())
    return (
        f"As seguintes classes possuem linha ou coluna zerada na matriz de teste temporal: {classes}. "
        "Em especial, se o perfil intermediario estiver zerado, isso indica que ele nao apareceu no conjunto de teste "
        "e deve ser interpretado como limitacao da amostra, nao como ausencia substantiva do perfil."
    )


def main() -> None:
    garantir_diretorios()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    base = resumo_base()
    cobertura = ler_csv_opcional(TABLES_DIR / "auditoria_cobertura_paises.csv")
    variaveis = ler_csv_opcional(TABLES_DIR / "auditoria_variaveis_candidatas.csv")
    variaveis_cluster = ler_csv_opcional(TABLES_DIR / "variaveis_usadas_clusterizacao.csv")
    variaveis_classificacao = ler_csv_opcional(TABLES_DIR / "variaveis_usadas_classificacao.csv")
    metricas_cluster = ler_csv_opcional(TABLES_DIR / "metricas_clusterizacao.csv")
    robustez = ler_csv_opcional(TABLES_DIR / "robustez_clusterizacao.csv")
    estabilidade = ler_csv_opcional(TABLES_DIR / "estabilidade_clusters_ari.csv")
    perfis = ler_csv_opcional(TABLES_DIR / "perfis_clusters_interpretados.csv")
    nomes = ler_csv_opcional(TABLES_DIR / "nomes_clusters_sugeridos.csv")
    paises_recentes = ler_csv_opcional(TABLES_DIR / "paises_por_cluster_ano_recente.csv")
    cluster_intermediario = ler_csv_opcional(TABLES_DIR / "cluster_intermediario_detalhado.csv")
    paises_indicadores_recentes = ler_csv_opcional(TABLES_DIR / "paises_ano_recente_indicadores_completos.csv")
    metricas_classificacao = ler_csv_opcional(TABLES_DIR / "metricas_classificacao.csv")
    metricas_sem_vazamento = ler_csv_opcional(TABLES_DIR / "metricas_classificacao_sem_vazamento.csv")
    comparacao_vazamento = ler_csv_opcional(TABLES_DIR / "comparacao_modelos_com_sem_vazamento.csv")
    diagnostico_matriz = ler_csv_opcional(TABLES_DIR / "diagnostico_matriz_confusao_artigo.csv")
    ablacao_classificacao = ler_csv_opcional(TABLES_DIR / "ablacao_variaveis_classificacao.csv")
    ablacao_clusterizacao = ler_csv_opcional(TABLES_DIR / "ablacao_variaveis_clusterizacao.csv")
    importancia = ler_csv_opcional(TABLES_DIR / "importancia_permutacao.csv")

    texto_completo = f"""# Relatorio de Resultados - MERCOSUL Descarbonizacao ML

## 1. Objetivo cientifico

Investigar trajetorias de descarbonizacao e transicao energetica em paises associados ao MERCOSUL usando mineracao de dados, clusterizacao, aprendizado de maquina supervisionado e interpretabilidade.

## 2. Fonte e descricao da base

A fonte principal e a Our World in Data Energy Dataset. A base processada possui `{base["qtd_observacoes"]}` observacoes, `{base["qtd_paises"]}` paises e cobertura temporal de `{base["ano_minimo"]}` a `{base["ano_maximo"]}`.

Paises analisados:

{", ".join(base["paises"])}

## 3. Cobertura temporal por pais

{tabela_markdown(cobertura, n=20)}

## 4. Variaveis candidatas e variaveis usadas

### Auditoria de variaveis candidatas

{tabela_markdown(variaveis, n=30)}

### Variaveis usadas na clusterizacao

{tabela_markdown(variaveis_cluster, n=30)}

### Variaveis usadas na classificacao

{tabela_markdown(variaveis_classificacao, n=30)}

## 5. Resultado da clusterizacao

{formatar_melhor_cluster(metricas_cluster)}

{tabela_markdown(metricas_cluster, n=12)}

## 6. Robustez da clusterizacao

A robustez foi avaliada variando KMeans, AgglomerativeClustering e GaussianMixture, com k entre 2 e 6 e multiplas seeds quando aplicavel.

{tabela_markdown(robustez, n=15)}

### Estabilidade entre execucoes

{tabela_markdown(estabilidade, n=15)}

## 7. Interpretacao automatica dos clusters

Os nomes sugeridos abaixo sao calculados a partir de desvios dos indicadores de cada cluster em relacao a media geral. Eles devem ser tratados como rotulos interpretativos, nao como categorias externas observadas.

{tabela_markdown(nomes, n=10)}

### Perfis interpretados

{tabela_markdown(perfis, n=10)}

### Paises por cluster no ano mais recente disponivel

{tabela_markdown(paises_recentes, n=20)}

### Cautela sobre o cluster intermediario

{resumir_cluster_intermediario(cluster_intermediario)}

Tabela detalhada salva em `results/tables/cluster_intermediario_detalhado.csv`.

{tabela_markdown(cluster_intermediario, n=12)}

### Interpretacao dos paises no ano mais recente com indicadores completos

A tabela abaixo organiza os indicadores centrais de cada pais no ano mais recente em que todos os indicadores exigidos estao disponiveis, junto ao perfil de transicao energetica atribuido pela clusterizacao. Quando nenhum ano possui todos os indicadores completos, a tabela registra essa limitacao explicitamente.

{tabela_markdown(paises_indicadores_recentes, n=20)}

## 8. Resultado da classificacao supervisionada

{formatar_melhor_classificacao(metricas_classificacao)}

{tabela_markdown(metricas_classificacao, n=10)}

Importante: o modelo supervisionado aprende a reproduzir perfis de transicao energetica identificados por clusterizacao nao supervisionada. Portanto, o alvo nao e uma classe externa real.

### Diagnostico da matriz de confusao

{resumir_diagnostico_matriz(diagnostico_matriz)}

{tabela_markdown(diagnostico_matriz, n=10)}

## 9. Validacao temporal sem vazamento

Na validacao sem vazamento, o imputador, o scaler e o KMeans sao ajustados apenas no treino ate 2022. Os anos de 2023 em diante sao atribuidos ao centroide treinado mais proximo, e os classificadores sao treinados apenas com o periodo de treino.

{formatar_melhor_classificacao(metricas_sem_vazamento)}

{tabela_markdown(metricas_sem_vazamento, n=10)}

### Comparacao com a validacao original

{tabela_markdown(comparacao_vazamento, n=10)}

## 10. Estudo de ablacao

O estudo de ablacao compara cenarios com todas as variaveis, sem o indice preliminar, apenas energia, apenas eletricidade e sem variaveis derivadas do projeto.

### Ablacao na classificacao

{tabela_markdown(ablacao_classificacao, n=15)}

### Ablacao na clusterizacao

{tabela_markdown(ablacao_clusterizacao, n=15)}

## 11. Importancia das variaveis

{tabela_markdown(importancia, n=15)}

## 12. Figuras geradas

As figuras cientificas foram salvas em `results/figures/`, incluindo PCA dos clusters, heatmap dos centroides, evolucao temporal dos clusters, evolucao de renovaveis, evolucao da intensidade de carbono, importancia por permutacao e matriz de confusao.

### Figuras recomendadas para o artigo

- `results/figures/pca_clusters.png`
- `results/figures/heatmap_clusters_pais_ano.png`
- `results/figures/heatmap_centroides_artigo.png`
- `results/figures/importancia_permutacao_top15.png`
- `results/figures/matriz_confusao_artigo.png`
- `results/figures/evolucao_intensidade_carbono_paises_chave.png`
- `results/figures/evolucao_renovaveis_paises_chave.png`

## 13. Limitacoes metodologicas

- Os rotulos supervisionados sao derivados da clusterizacao e nao representam classes externas observadas.
- A qualidade dos clusters depende das variaveis selecionadas, do tratamento de ausencias e do recorte temporal.
- O desempenho perfeito ou muito alto na classificacao deve ser interpretado como capacidade de reproduzir rotulos derivados, nao como prova causal de descarbonizacao.
- Algumas variaveis possuem lacunas historicas, especialmente nos anos mais antigos.
- O indice de transicao preliminar e uma variavel composta; por isso, seus efeitos devem ser discutidos com apoio do estudo de ablacao.
- O cluster intermediario e restrito a uma trajetoria nacional especifica quando associado a apenas um pais, devendo ser interpretado com cautela.
- A matriz de confusao da validacao temporal pode nao conter o perfil intermediario no conjunto de teste, produzindo linha ou coluna zerada para essa classe.
- A base OWID pode sofrer revisoes, entao execucoes futuras podem alterar pequenos valores e metricas.

## 14. Proximos passos

- Revisar as figuras para selecionar as que entram no artigo.
- Priorizar resultados sem vazamento na narrativa metodologica.
- Usar os nomes de clusters como hipoteses interpretativas e validar com literatura energetica regional.
- Comparar trajetorias nacionais recentes com eventos de politica energetica e contexto economico.
- Preparar tabelas finais em LaTeX a partir de `results/latex/`.

## 15. Parecer automatico

{avaliar_viabilidade(metricas_classificacao, metricas_cluster, metricas_sem_vazamento)}
"""

    texto_viabilidade = f"""# Relatorio de Viabilidade Experimental

## Objetivo

Consolidar evidencias iniciais para decidir se o experimento pode avancar para escrita cientifica.

## Resumo dos resultados

- Base: {base["qtd_observacoes"]} observacoes, {base["qtd_paises"]} paises, periodo {base["ano_minimo"]}-{base["ano_maximo"]}.
- Clusterizacao: {formatar_melhor_cluster(metricas_cluster)}
- Classificacao original: {formatar_melhor_classificacao(metricas_classificacao)}
- Validacao sem vazamento: {formatar_melhor_classificacao(metricas_sem_vazamento)}

## Parecer automatico

{avaliar_viabilidade(metricas_classificacao, metricas_cluster, metricas_sem_vazamento)}

## Observacao metodologica

Os modelos supervisionados reproduzem rotulos derivados de clusterizacao. A narrativa do artigo deve apresentar a classificacao como reproducao de perfis identificados, nao como predicao de uma classe externa real.

## Tabelas principais

### Clusterizacao

{tabela_markdown(metricas_cluster, n=10)}

### Classificacao sem vazamento

{tabela_markdown(metricas_sem_vazamento, n=10)}

### Interpretacao dos clusters

{tabela_markdown(nomes, n=10)}
"""

    caminho_completo = REPORTS_DIR / "relatorio_resultados_completo.md"
    caminho_viabilidade = REPORTS_DIR / "relatorio_viabilidade.md"
    caminho_completo.write_text(texto_completo, encoding="utf-8")
    caminho_viabilidade.write_text(texto_viabilidade, encoding="utf-8")

    print(f"Relatorio salvo: {caminho_viabilidade}")
    print(f"Relatorio completo salvo: {caminho_completo}")


if __name__ == "__main__":
    main()
