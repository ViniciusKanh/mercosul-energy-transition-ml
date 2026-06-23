# -*- coding: utf-8 -*-

"""
Script 09 - Robustez da clusterizacao.

Objetivo:
Avaliar se os agrupamentos sao estaveis ao variar algoritmo, numero de clusters
e sementes aleatorias quando aplicavel.

Execucao:
python scripts/09_robustez_clusterizacao.py
"""

from pathlib import Path
from itertools import combinations
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.config import PROCESSED_DIR, TABLES_DIR, VARIAVEIS_CLUSTER_PREFERENCIAIS  # noqa: E402
from utils.io_utils import garantir_diretorios, salvar_csv, verificar_arquivo  # noqa: E402

warnings.filterwarnings("ignore")

ANO_MINIMO_ANALISE = 2000
SEEDS = [0, 1, 7, 21, 42, 99, 123, 2024]
VALORES_K = range(2, 7)


def selecionar_variaveis(df: pd.DataFrame, limite_ausentes: float = 0.45) -> list[str]:
    """Seleciona variaveis com cobertura suficiente para robustez."""
    selecionadas = []

    for coluna in VARIAVEIS_CLUSTER_PREFERENCIAIS:
        if coluna not in df.columns:
            continue

        serie = pd.to_numeric(df[coluna], errors="coerce")
        if serie.notna().mean() >= (1 - limite_ausentes):
            selecionadas.append(coluna)

    return selecionadas


def gerar_labels(algoritmo: str, k: int, seed: int | None, X_proc: np.ndarray) -> np.ndarray:
    """Executa um algoritmo de clusterizacao e retorna seus labels."""
    if algoritmo == "kmeans":
        modelo = KMeans(n_clusters=k, random_state=seed, n_init=30)
        return modelo.fit_predict(X_proc)

    if algoritmo == "agglomerative":
        modelo = AgglomerativeClustering(n_clusters=k)
        return modelo.fit_predict(X_proc)

    if algoritmo == "gmm":
        modelo = GaussianMixture(n_components=k, random_state=seed, n_init=5)
        return modelo.fit_predict(X_proc)

    raise ValueError(f"Algoritmo desconhecido: {algoritmo}")


def calcular_metricas(X_proc: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Calcula metricas internas de qualidade dos clusters."""
    if len(set(labels)) < 2:
        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
        }

    return {
        "silhouette": silhouette_score(X_proc, labels),
        "davies_bouldin": davies_bouldin_score(X_proc, labels),
        "calinski_harabasz": calinski_harabasz_score(X_proc, labels),
    }


def main() -> None:
    garantir_diretorios()

    caminho_entrada = PROCESSED_DIR / "02_dataset_atributos.csv"
    verificar_arquivo(caminho_entrada)

    df = pd.read_csv(caminho_entrada)
    df = df[df["year"] >= ANO_MINIMO_ANALISE].copy()
    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    variaveis = selecionar_variaveis(df)
    if len(variaveis) < 4:
        raise ValueError("Poucas variaveis disponiveis para robustez da clusterizacao.")

    X = df[variaveis].apply(pd.to_numeric, errors="coerce")
    preprocessador = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_proc = preprocessador.fit_transform(X)

    metricas_registros = []
    distribuicao_registros = []
    labels_por_configuracao: dict[tuple[str, int], list[dict[str, object]]] = {}

    for k in VALORES_K:
        for algoritmo in ["kmeans", "agglomerative", "gmm"]:
            seeds = SEEDS if algoritmo in {"kmeans", "gmm"} else [None]

            for seed in seeds:
                seed_rotulo = "na" if seed is None else str(seed)
                modelo_id = f"{algoritmo}_k{k}_seed{seed_rotulo}"

                try:
                    labels = gerar_labels(algoritmo, k, seed, X_proc)
                    metricas = calcular_metricas(X_proc, labels)
                except Exception as erro:
                    print(f"Falha em {modelo_id}: {erro}")
                    continue

                metricas_registros.append(
                    {
                        "modelo_id": modelo_id,
                        "algoritmo": algoritmo,
                        "k": k,
                        "seed": seed,
                        "qtd_observacoes": len(df),
                        "qtd_variaveis": len(variaveis),
                        **metricas,
                    }
                )

                labels_por_configuracao.setdefault((algoritmo, k), []).append(
                    {"modelo_id": modelo_id, "seed": seed, "labels": labels}
                )

                distribuicao = (
                    df.assign(cluster=labels)
                    .groupby(["country", "cluster"])
                    .agg(
                        qtd_observacoes=("year", "size"),
                        ano_minimo=("year", "min"),
                        ano_maximo=("year", "max"),
                    )
                    .reset_index()
                )
                distribuicao["modelo_id"] = modelo_id
                distribuicao["algoritmo"] = algoritmo
                distribuicao["k"] = k
                distribuicao["seed"] = seed
                distribuicao_registros.append(distribuicao)

    estabilidade_registros = []
    for (algoritmo, k), execucoes in labels_por_configuracao.items():
        if len(execucoes) < 2:
            estabilidade_registros.append(
                {
                    "algoritmo": algoritmo,
                    "k": k,
                    "modelo_a": execucoes[0]["modelo_id"],
                    "modelo_b": None,
                    "seed_a": execucoes[0]["seed"],
                    "seed_b": None,
                    "adjusted_rand_index": np.nan,
                    "normalized_mutual_info": np.nan,
                    "observacao": "Algoritmo deterministico ou sem multiplas seeds.",
                }
            )
            continue

        for a, b in combinations(execucoes, 2):
            estabilidade_registros.append(
                {
                    "algoritmo": algoritmo,
                    "k": k,
                    "modelo_a": a["modelo_id"],
                    "modelo_b": b["modelo_id"],
                    "seed_a": a["seed"],
                    "seed_b": b["seed"],
                    "adjusted_rand_index": adjusted_rand_score(a["labels"], b["labels"]),
                    "normalized_mutual_info": normalized_mutual_info_score(a["labels"], b["labels"]),
                    "observacao": "Comparacao par a par entre execucoes.",
                }
            )

    metricas_df = pd.DataFrame(metricas_registros).sort_values(
        ["silhouette", "davies_bouldin"],
        ascending=[False, True],
    )
    estabilidade_df = pd.DataFrame(estabilidade_registros)
    distribuicao_df = pd.concat(distribuicao_registros, ignore_index=True)

    salvar_csv(metricas_df, TABLES_DIR / "robustez_clusterizacao.csv")
    salvar_csv(estabilidade_df, TABLES_DIR / "estabilidade_clusters_ari.csv")
    salvar_csv(distribuicao_df, TABLES_DIR / "distribuicao_clusters_por_modelo.csv")

    print("\nRobustez da clusterizacao concluida.")
    print(metricas_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
