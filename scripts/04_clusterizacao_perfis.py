# -*- coding: utf-8 -*-

"""
Script 04 - Clusterização dos perfis de transição energética.

Objetivo:
Construir uma variável-alvo inicial por agrupamento não supervisionado.

Execução:
python scripts/04_clusterizacao_perfis.py
python scripts/04_clusterizacao_perfis.py --ano-min 2000 --ano-max 2023
"""

from pathlib import Path
import argparse
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.config import (  # noqa: E402
    PROCESSED_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    VARIAVEIS_CLUSTER_PREFERENCIAIS,
    RANDOM_STATE,
)
from utils.io_utils import garantir_diretorios, salvar_csv, verificar_arquivo  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)


def selecionar_variaveis_cluster(df: pd.DataFrame, limite_ausentes: float = 0.45) -> list[str]:
    """Seleciona variáveis existentes com cobertura mínima para clusterização."""
    selecionadas = []

    for coluna in VARIAVEIS_CLUSTER_PREFERENCIAIS:
        if coluna in df.columns:
            serie = pd.to_numeric(df[coluna], errors="coerce")
            if serie.notna().mean() >= (1 - limite_ausentes):
                selecionadas.append(coluna)

    return selecionadas


def rotular_clusters_por_indice(df: pd.DataFrame, coluna_cluster: str) -> dict:
    """
    Nomeia clusters a partir do índice preliminar de transição.

    A regra é simples e auditável. A interpretação final deve ser confirmada
    com os perfis médios salvos em results/tables/perfis_clusters.csv.
    """
    perfil = (
        df.groupby(coluna_cluster)
        .agg(
            indice_medio=("indice_transicao_preliminar", "mean"),
            fossil_medio=("fossil_share_energy", "mean"),
            renovavel_medio=("renewables_share_energy", "mean"),
        )
        .reset_index()
        .sort_values("indice_medio")
    )

    clusters_ordenados = perfil[coluna_cluster].tolist()
    n = len(clusters_ordenados)

    if n == 2:
        nomes = ["baixa_transicao", "transicao_avancada"]
    elif n == 3:
        nomes = ["baixa_transicao", "transicao_intermediaria", "transicao_avancada"]
    elif n == 4:
        nomes = ["baixa_transicao", "transicao_intermediaria", "transicao_avancada", "transicao_muito_avancada"]
    else:
        nomes = [f"perfil_{i + 1}" for i in range(n)]

    return dict(zip(clusters_ordenados, nomes))


def main() -> None:
    parser = argparse.ArgumentParser(description="Clusterização dos perfis de descarbonização.")
    parser.add_argument("--ano-min", type=int, default=2000, help="Ano inicial do recorte.")
    parser.add_argument("--ano-max", type=int, default=None, help="Ano final do recorte.")
    args = parser.parse_args()

    garantir_diretorios()

    caminho_entrada = PROCESSED_DIR / "02_dataset_atributos.csv"
    verificar_arquivo(caminho_entrada)

    df = pd.read_csv(caminho_entrada)
    df = df[df["year"] >= args.ano_min].copy()

    if args.ano_max is not None:
        df = df[df["year"] <= args.ano_max].copy()

    variaveis = selecionar_variaveis_cluster(df)

    if len(variaveis) < 4:
        raise ValueError(
            "Poucas variáveis disponíveis para clusterização. "
            "Revise a auditoria de dados e o recorte temporal."
        )

    print("Variáveis usadas na clusterização:")
    print(variaveis)

    X = df[variaveis].apply(pd.to_numeric, errors="coerce")

    preprocessador = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_proc = preprocessador.fit_transform(X)

    registros_metricas = []
    resultados_labels = {}

    for k in range(2, 6):
        modelos = {
            f"kmeans_k{k}": KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20),
            f"agglomerative_k{k}": AgglomerativeClustering(n_clusters=k),
            f"gmm_k{k}": GaussianMixture(n_components=k, random_state=RANDOM_STATE),
        }

        for nome, modelo in modelos.items():
            labels = modelo.fit_predict(X_proc)

            if len(set(labels)) < 2:
                continue

            registros_metricas.append(
                {
                    "modelo_cluster": nome,
                    "k": k,
                    "silhouette": silhouette_score(X_proc, labels),
                    "davies_bouldin": davies_bouldin_score(X_proc, labels),
                    "calinski_harabasz": calinski_harabasz_score(X_proc, labels),
                }
            )
            resultados_labels[nome] = labels

    metricas = pd.DataFrame(registros_metricas).sort_values(
        ["silhouette", "davies_bouldin"],
        ascending=[False, True],
    )

    if metricas.empty:
        raise ValueError("Nenhum resultado de clusterização foi gerado.")

    melhor_modelo = metricas.iloc[0]["modelo_cluster"]
    labels_melhor = resultados_labels[melhor_modelo]

    df["cluster_transicao"] = labels_melhor
    mapa_nomes = rotular_clusters_por_indice(df, "cluster_transicao")
    df["perfil_transicao"] = df["cluster_transicao"].map(mapa_nomes)
    df["modelo_cluster_escolhido"] = melhor_modelo

    colunas_perfil = [
        "cluster_transicao",
        "perfil_transicao",
        "country",
        "year",
    ] + variaveis

    perfil = (
        df[colunas_perfil]
        .groupby(["cluster_transicao", "perfil_transicao"])
        .agg(
            qtd_observacoes=("year", "size"),
            ano_minimo=("year", "min"),
            ano_maximo=("year", "max"),
            qtd_paises=("country", "nunique"),
            **{f"media_{v}": (v, "mean") for v in variaveis},
        )
        .reset_index()
        .sort_values("cluster_transicao")
    )

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_proc)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=df["cluster_transicao"], alpha=0.75)
    ax.set_title("PCA dos perfis de transição energética")
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.grid(True, alpha=0.3)
    legenda = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legenda)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pca_clusters_transicao.png", dpi=300)
    plt.close(fig)

    salvar_csv(metricas, TABLES_DIR / "metricas_clusterizacao.csv")
    salvar_csv(perfil, TABLES_DIR / "perfis_clusters.csv")
    salvar_csv(pd.DataFrame({"variavel_usada": variaveis}), TABLES_DIR / "variaveis_usadas_clusterizacao.csv")
    salvar_csv(df, PROCESSED_DIR / "03_dataset_com_clusters.csv")

    print("\nClusterização concluída.")
    print(f"Melhor modelo pelo critério inicial: {melhor_modelo}")
    print(metricas.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
