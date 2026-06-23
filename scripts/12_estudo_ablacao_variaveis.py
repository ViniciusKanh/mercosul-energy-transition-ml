# -*- coding: utf-8 -*-

"""
Script 12 - Estudo de ablacao de variaveis.

Objetivo:
Verificar a sensibilidade dos resultados a conjuntos alternativos de variaveis,
especialmente a remocao do indice composto de transicao preliminar.

Execucao:
python scripts/12_estudo_ablacao_variaveis.py
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.config import PROCESSED_DIR, TABLES_DIR, VARIAVEIS_CLUSTER_PREFERENCIAIS, RANDOM_STATE  # noqa: E402
from utils.io_utils import garantir_diretorios, salvar_csv, verificar_arquivo  # noqa: E402
from utils.ml_utils import calcular_metricas_classificacao  # noqa: E402

warnings.filterwarnings("ignore")

ANO_MINIMO_ANALISE = 2000

VARIAVEIS_DERIVADAS_PROJETO = {
    "razao_renovavel_fossil_energy",
    "variacao_co2_pct",
    "variacao_renewables_share_energy_pp",
    "media_movel_co2_3anos",
    "tendencia_renovaveis_5anos",
    "indice_transicao_preliminar",
}


def variaveis_disponiveis(df: pd.DataFrame, limite_ausentes: float = 0.45) -> list[str]:
    """Seleciona variaveis modelaveis com cobertura minima."""
    selecionadas = []
    for coluna in VARIAVEIS_CLUSTER_PREFERENCIAIS:
        if coluna not in df.columns:
            continue
        serie = pd.to_numeric(df[coluna], errors="coerce")
        if serie.notna().mean() >= (1 - limite_ausentes):
            selecionadas.append(coluna)
    return selecionadas


def montar_cenarios(variaveis: list[str]) -> dict[str, list[str]]:
    """Cria os cenarios de ablacao solicitados."""
    energia = [
        v
        for v in variaveis
        if (
            "energy" in v
            or v in {"energy_per_capita", "energy_per_gdp", "co2_per_unit_energy"}
        )
        and "_elec" not in v
        and "electricity" not in v
    ]
    eletricidade = [
        v
        for v in variaveis
        if "_elec" in v or "electricity" in v or v == "carbon_intensity_elec"
    ]
    sem_derivadas = [v for v in variaveis if v not in VARIAVEIS_DERIVADAS_PROJETO]

    return {
        "A_todas_variaveis": variaveis,
        "B_sem_indice_transicao_preliminar": [v for v in variaveis if v != "indice_transicao_preliminar"],
        "C_apenas_variaveis_energia": energia,
        "D_apenas_variaveis_eletricidade": eletricidade,
        "E_sem_variaveis_derivadas": sem_derivadas,
    }


def split_temporal_ou_fallback(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Usa divisao temporal 2023+ quando possivel; caso contrario, usa fallback estratificado."""
    treino = df[df["year"] <= 2022].copy()
    teste = df[df["year"] >= 2023].copy()

    if treino["perfil_transicao"].nunique() < 2 or teste["perfil_transicao"].nunique() < 2 or len(teste) < 10:
        treino, teste = train_test_split(
            df,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=df["perfil_transicao"] if df["perfil_transicao"].nunique() > 1 else None,
        )
        return treino, teste, "estratificada_fallback"

    return treino, teste, "temporal_treino_ate_2022_teste_2023_em_diante"


def carregar_modelos(qtd_treino: int) -> dict[str, object]:
    """Modelos leves para comparar cenarios de variaveis."""
    n_vizinhos = max(1, min(5, qtd_treino))
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
        ),
        "knn": KNeighborsClassifier(n_neighbors=n_vizinhos),
    }


def avaliar_classificacao(df_clusterizado: pd.DataFrame, cenarios: dict[str, list[str]]) -> pd.DataFrame:
    """Avalia classificadores para reproduzir os perfis existentes sob cada cenario."""
    registros = []
    dados = df_clusterizado.dropna(subset=["perfil_transicao"]).copy()
    treino, teste, estrategia = split_temporal_ou_fallback(dados)
    modelos = carregar_modelos(len(treino))

    for cenario, variaveis in cenarios.items():
        variaveis = [v for v in variaveis if v in dados.columns]
        if len(variaveis) < 2:
            registros.append(
                {
                    "cenario": cenario,
                    "modelo": "nao_executado",
                    "qtd_variaveis": len(variaveis),
                    "variaveis": "; ".join(variaveis),
                    "observacao": "Cenario com menos de duas variaveis disponiveis.",
                }
            )
            continue

        X_train = treino[variaveis].apply(pd.to_numeric, errors="coerce")
        y_train = treino["perfil_transicao"]
        X_test = teste[variaveis].apply(pd.to_numeric, errors="coerce")
        y_test = teste["perfil_transicao"]

        preprocessador = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    variaveis,
                )
            ]
        )

        for nome_modelo, modelo in modelos.items():
            pipeline = Pipeline(
                steps=[
                    ("preprocessador", preprocessador),
                    ("modelo", modelo),
                ]
            )

            try:
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                metricas = calcular_metricas_classificacao(y_test, y_pred, nome_modelo)
                metricas.update(
                    {
                        "cenario": cenario,
                        "qtd_variaveis": len(variaveis),
                        "variaveis": "; ".join(variaveis),
                        "estrategia_validacao": estrategia,
                        "qtd_treino": len(treino),
                        "qtd_teste": len(teste),
                        "observacao": "Classificacao dos rotulos derivados da clusterizacao original.",
                    }
                )
                registros.append(metricas)
            except Exception as erro:
                registros.append(
                    {
                        "cenario": cenario,
                        "modelo": nome_modelo,
                        "qtd_variaveis": len(variaveis),
                        "variaveis": "; ".join(variaveis),
                        "observacao": f"Falha ao treinar modelo: {erro}",
                    }
                )

    return pd.DataFrame(registros)


def avaliar_clusterizacao(df_atributos: pd.DataFrame, cenarios: dict[str, list[str]]) -> pd.DataFrame:
    """Avalia qualidade de KMeans em cada cenario para k de 2 a 6."""
    registros = []
    dados = df_atributos[df_atributos["year"] >= ANO_MINIMO_ANALISE].copy()

    for cenario, variaveis in cenarios.items():
        variaveis = [v for v in variaveis if v in dados.columns]
        if len(variaveis) < 2:
            registros.append(
                {
                    "cenario": cenario,
                    "k": np.nan,
                    "qtd_variaveis": len(variaveis),
                    "variaveis": "; ".join(variaveis),
                    "observacao": "Cenario com menos de duas variaveis disponiveis.",
                }
            )
            continue

        X = dados[variaveis].apply(pd.to_numeric, errors="coerce")
        preprocessador = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        X_proc = preprocessador.fit_transform(X)

        for k in range(2, 7):
            try:
                labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50).fit_predict(X_proc)
                registros.append(
                    {
                        "cenario": cenario,
                        "k": k,
                        "qtd_variaveis": len(variaveis),
                        "variaveis": "; ".join(variaveis),
                        "silhouette": silhouette_score(X_proc, labels),
                        "davies_bouldin": davies_bouldin_score(X_proc, labels),
                        "calinski_harabasz": calinski_harabasz_score(X_proc, labels),
                        "observacao": "KMeans ajustado para o conjunto de variaveis do cenario.",
                    }
                )
            except Exception as erro:
                registros.append(
                    {
                        "cenario": cenario,
                        "k": k,
                        "qtd_variaveis": len(variaveis),
                        "variaveis": "; ".join(variaveis),
                        "observacao": f"Falha na clusterizacao: {erro}",
                    }
                )

    return pd.DataFrame(registros)


def main() -> None:
    garantir_diretorios()

    caminho_atributos = PROCESSED_DIR / "02_dataset_atributos.csv"
    caminho_clusterizado = PROCESSED_DIR / "03_dataset_com_clusters.csv"
    verificar_arquivo(caminho_atributos)
    verificar_arquivo(caminho_clusterizado)

    df_atributos = pd.read_csv(caminho_atributos)
    df_clusterizado = pd.read_csv(caminho_clusterizado)

    variaveis = variaveis_disponiveis(df_atributos[df_atributos["year"] >= ANO_MINIMO_ANALISE].copy())
    cenarios = montar_cenarios(variaveis)

    ablacao_classificacao = avaliar_classificacao(df_clusterizado, cenarios)
    ablacao_clusterizacao = avaliar_clusterizacao(df_atributos, cenarios)

    ablacao_classificacao = ablacao_classificacao.sort_values(
        ["cenario", "f1_macro", "balanced_accuracy"],
        ascending=[True, False, False],
        na_position="last",
    )
    ablacao_clusterizacao = ablacao_clusterizacao.sort_values(
        ["cenario", "silhouette", "davies_bouldin"],
        ascending=[True, False, True],
        na_position="last",
    )

    salvar_csv(ablacao_classificacao, TABLES_DIR / "ablacao_variaveis_classificacao.csv")
    salvar_csv(ablacao_clusterizacao, TABLES_DIR / "ablacao_variaveis_clusterizacao.csv")

    print("\nEstudo de ablacao concluido.")
    print(ablacao_classificacao.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
