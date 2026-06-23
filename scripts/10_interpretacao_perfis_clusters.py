# -*- coding: utf-8 -*-

"""
Script 10 - Interpretacao automatica dos perfis de clusters.

Objetivo:
Transformar medias de clusters em uma tabela interpretavel, com nomes sugeridos
por regras objetivas baseadas em desvios em relacao a media geral.

Execucao:
python scripts/10_interpretacao_perfis_clusters.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.config import PROCESSED_DIR, TABLES_DIR, VARIAVEIS_CLUSTER_PREFERENCIAIS  # noqa: E402
from utils.io_utils import garantir_diretorios, salvar_csv, verificar_arquivo  # noqa: E402


INDICADORES_BAIXO_CARBONO = [
    "renewables_share_energy",
    "low_carbon_share_energy",
    "renewables_share_elec",
    "low_carbon_share_elec",
    "indice_transicao_preliminar",
]

INDICADORES_FOSSEIS = [
    "fossil_share_energy",
    "fossil_share_elec",
    "carbon_intensity_elec",
    "co2_per_unit_energy",
]


def carregar_variaveis(df: pd.DataFrame) -> list[str]:
    """Carrega variaveis usadas na clusterizacao quando a tabela existir."""
    caminho_variaveis = TABLES_DIR / "variaveis_usadas_clusterizacao.csv"

    if caminho_variaveis.exists():
        variaveis = pd.read_csv(caminho_variaveis)["variavel_usada"].dropna().tolist()
    else:
        variaveis = [v for v in VARIAVEIS_CLUSTER_PREFERENCIAIS if v in df.columns]

    return [v for v in variaveis if v in df.columns]


def media_lista(linha: pd.Series, colunas: list[str]) -> float:
    """Calcula media de colunas existentes em uma linha."""
    existentes = [c for c in colunas if c in linha.index and pd.notna(linha[c])]
    if not existentes:
        return np.nan
    return float(linha[existentes].mean())


def resumir_extremos(linha_z: pd.Series, limite: float = 0.30) -> tuple[str, str]:
    """Resume variaveis acima e abaixo da media geral."""
    ordenada = linha_z.dropna().sort_values(ascending=False)
    acima = ordenada[ordenada >= limite].head(4)
    abaixo = ordenada[ordenada <= -limite].sort_values().head(4)

    texto_acima = "; ".join([f"{idx} ({valor:.2f} dp)" for idx, valor in acima.items()])
    texto_abaixo = "; ".join([f"{idx} ({valor:.2f} dp)" for idx, valor in abaixo.items()])

    return texto_acima or "Sem desvios positivos relevantes", texto_abaixo or "Sem desvios negativos relevantes"


def nomear_cluster(score_baixo_carbono: float, score_fossil: float, score_indice: float) -> tuple[str, str]:
    """Aplica regras objetivas para sugerir nomes de perfis."""
    if pd.notna(score_fossil) and score_fossil >= 0.30 and (pd.isna(score_baixo_carbono) or score_baixo_carbono <= 0.10):
        return (
            "Perfil fossil-intensivo",
            "Indicadores fosseis e/ou intensidade de carbono acima da media, com baixo carbono sem destaque positivo.",
        )

    if pd.notna(score_baixo_carbono) and score_baixo_carbono >= 0.30 and (pd.isna(score_fossil) or score_fossil <= 0.10):
        return (
            "Perfil renovavel/baixo carbono",
            "Indicadores renovaveis e de baixo carbono acima da media, sem destaque fossil equivalente.",
        )

    if pd.notna(score_indice) and score_indice >= 0.30:
        return (
            "Perfil de transicao relativamente avancada",
            "Indice preliminar de transicao acima da media, mas com sinais mistos nos demais indicadores.",
        )

    if pd.notna(score_indice) and score_indice <= -0.30:
        return (
            "Perfil de baixa transicao relativa",
            "Indice preliminar de transicao abaixo da media, mas sem padrao fossil exclusivo suficiente.",
        )

    return (
        "Perfil intermediario",
        "Indicadores proximos da media geral ou combinacao mista de sinais.",
    )


def main() -> None:
    garantir_diretorios()

    caminho_dataset = PROCESSED_DIR / "03_dataset_com_clusters.csv"
    caminho_perfis = TABLES_DIR / "perfis_clusters.csv"
    verificar_arquivo(caminho_dataset)
    verificar_arquivo(caminho_perfis)

    df = pd.read_csv(caminho_dataset)
    perfis_originais = pd.read_csv(caminho_perfis)

    if "cluster_transicao" not in df.columns:
        raise ValueError("A coluna cluster_transicao nao foi encontrada no dataset clusterizado.")

    variaveis = carregar_variaveis(df)
    if len(variaveis) < 4:
        raise ValueError("Poucas variaveis disponiveis para interpretar os clusters.")

    dados_numericos = df[variaveis].apply(pd.to_numeric, errors="coerce")
    medias_cluster = dados_numericos.groupby(df["cluster_transicao"]).mean()
    media_geral = dados_numericos.mean()
    desvio_geral = dados_numericos.std(ddof=0).replace(0, np.nan)
    z_cluster = (medias_cluster - media_geral) / desvio_geral

    registros_nomes = []
    registros_perfis = []

    for cluster, linha_z in z_cluster.iterrows():
        score_baixo_carbono = media_lista(linha_z, INDICADORES_BAIXO_CARBONO)
        score_fossil = media_lista(linha_z, INDICADORES_FOSSEIS)
        score_indice = linha_z.get("indice_transicao_preliminar", np.nan)
        nome_sugerido, regra = nomear_cluster(score_baixo_carbono, score_fossil, score_indice)
        acima, abaixo = resumir_extremos(linha_z)

        registros_nomes.append(
            {
                "cluster_transicao": cluster,
                "nome_sugerido": nome_sugerido,
                "score_baixo_carbono_z": score_baixo_carbono,
                "score_fossil_z": score_fossil,
                "score_indice_transicao_z": score_indice,
                "regra_aplicada": regra,
                "variaveis_acima_media": acima,
                "variaveis_abaixo_media": abaixo,
            }
        )

        registro = {
            "cluster_transicao": cluster,
            "nome_sugerido": nome_sugerido,
            "caracteristica_principal": regra,
            "variaveis_acima_media": acima,
            "variaveis_abaixo_media": abaixo,
        }
        for variavel in variaveis:
            registro[f"media_{variavel}"] = medias_cluster.loc[cluster, variavel]
            registro[f"desvio_padronizado_{variavel}"] = z_cluster.loc[cluster, variavel]
        registros_perfis.append(registro)

    nomes_df = pd.DataFrame(registros_nomes).sort_values("cluster_transicao")
    perfis_interpretados = pd.DataFrame(registros_perfis).sort_values("cluster_transicao")
    perfis_interpretados = perfis_interpretados.merge(
        perfis_originais,
        on="cluster_transicao",
        how="left",
        suffixes=("", "_original"),
    )

    paises_recentes = (
        df.sort_values(["country", "year"])
        .groupby("country")
        .tail(1)[["country", "year", "cluster_transicao", "perfil_transicao"]]
        .merge(nomes_df[["cluster_transicao", "nome_sugerido"]], on="cluster_transicao", how="left")
        .sort_values(["cluster_transicao", "country"])
        .reset_index(drop=True)
    )

    salvar_csv(perfis_interpretados, TABLES_DIR / "perfis_clusters_interpretados.csv")
    salvar_csv(nomes_df, TABLES_DIR / "nomes_clusters_sugeridos.csv")
    salvar_csv(paises_recentes, TABLES_DIR / "paises_por_cluster_ano_recente.csv")

    print("\nInterpretacao dos perfis de clusters concluida.")
    print(nomes_df.to_string(index=False))


if __name__ == "__main__":
    main()
