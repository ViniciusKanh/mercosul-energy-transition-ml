# -*- coding: utf-8 -*-

"""
Script 08 - Validacao temporal sem vazamento metodologico.

Objetivo:
Recriar os rotulos de cluster usando apenas dados de treino temporal e avaliar
classificadores em dados futuros, evitando ajuste de transformadores com
informacao do periodo de teste.

Execucao:
python scripts/08_validacao_temporal_sem_vazamento.py
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.config import (  # noqa: E402
    PROCESSED_DIR,
    TABLES_DIR,
    VARIAVEIS_CLUSTER_PREFERENCIAIS,
    RANDOM_STATE,
)
from utils.io_utils import garantir_diretorios, salvar_csv, verificar_arquivo  # noqa: E402
from utils.ml_utils import calcular_metricas_classificacao  # noqa: E402

warnings.filterwarnings("ignore")

ANO_LIMITE_TREINO = 2022
ANO_INICIO_TESTE = 2023
ANO_MINIMO_ANALISE = 2000
K_CLUSTERS = 3


def selecionar_variaveis_treino(df_treino: pd.DataFrame, limite_ausentes: float = 0.45) -> list[str]:
    """Seleciona variaveis com cobertura suficiente usando apenas o treino."""
    selecionadas = []

    for coluna in VARIAVEIS_CLUSTER_PREFERENCIAIS:
        if coluna not in df_treino.columns:
            continue

        serie = pd.to_numeric(df_treino[coluna], errors="coerce")
        if serie.notna().mean() >= (1 - limite_ausentes):
            selecionadas.append(coluna)

    return selecionadas


def nomear_clusters_por_treino(
    df_treino: pd.DataFrame,
    labels_treino: np.ndarray,
) -> dict[int, str]:
    """
    Nomeia clusters por uma regra auditavel baseada no treino.

    A ordenacao usa o indice preliminar quando disponivel. Se ele estiver
    ausente, usa uma combinacao simples de renovaveis, baixo carbono e menor
    intensidade fossil.
    """
    dados = df_treino.copy()
    dados["cluster_sem_vazamento"] = labels_treino

    if "indice_transicao_preliminar" in dados.columns:
        score = (
            dados.groupby("cluster_sem_vazamento")["indice_transicao_preliminar"]
            .mean()
            .rename("score_transicao")
        )
    else:
        componentes = []
        for coluna in ["renewables_share_energy", "low_carbon_share_elec"]:
            if coluna in dados.columns:
                componentes.append(dados.groupby("cluster_sem_vazamento")[coluna].mean())
        for coluna in ["fossil_share_energy", "carbon_intensity_elec"]:
            if coluna in dados.columns:
                componentes.append(-dados.groupby("cluster_sem_vazamento")[coluna].mean())

        if componentes:
            score = pd.concat(componentes, axis=1).mean(axis=1).rename("score_transicao")
        else:
            score = pd.Series(
                range(dados["cluster_sem_vazamento"].nunique()),
                index=sorted(dados["cluster_sem_vazamento"].unique()),
                name="score_transicao",
            )

    clusters_ordenados = score.sort_values().index.tolist()

    if len(clusters_ordenados) == 2:
        nomes = ["baixa_transicao", "transicao_avancada"]
    elif len(clusters_ordenados) == 3:
        nomes = ["baixa_transicao", "transicao_intermediaria", "transicao_avancada"]
    else:
        nomes = [f"perfil_{i + 1}" for i in range(len(clusters_ordenados))]

    return {int(cluster): nome for cluster, nome in zip(clusters_ordenados, nomes)}


def carregar_modelos(qtd_treino: int) -> dict[str, object]:
    """Define modelos supervisionados obrigatorios e sem dependencias opcionais."""
    n_vizinhos = max(1, min(5, qtd_treino))

    return {
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "knn": KNeighborsClassifier(n_neighbors=n_vizinhos),
        "svm_rbf": SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
        ),
    }


def comparar_com_validacao_original(metricas_sem_vazamento: pd.DataFrame) -> pd.DataFrame:
    """Compara o melhor resultado novo com a classificacao original, se existir."""
    registros = []

    caminho_original = TABLES_DIR / "metricas_classificacao.csv"
    if caminho_original.exists():
        metricas_originais = pd.read_csv(caminho_original)
        if not metricas_originais.empty:
            melhor_original = metricas_originais.sort_values(
                ["f1_macro", "balanced_accuracy"],
                ascending=[False, False],
            ).iloc[0]
            registros.append(
                {
                    "cenario": "validacao_original_com_rotulos_clusterizados_em_toda_amostra",
                    "modelo": melhor_original.get("modelo"),
                    "accuracy": melhor_original.get("accuracy"),
                    "balanced_accuracy": melhor_original.get("balanced_accuracy"),
                    "f1_macro": melhor_original.get("f1_macro"),
                    "f1_weighted": melhor_original.get("f1_weighted"),
                    "observacao": "Referencia historica; pode conter vazamento na construcao dos rotulos.",
                }
            )

    if not metricas_sem_vazamento.empty:
        melhor_sem_vazamento = metricas_sem_vazamento.sort_values(
            ["f1_macro", "balanced_accuracy"],
            ascending=[False, False],
        ).iloc[0]
        registros.append(
            {
                "cenario": "validacao_temporal_sem_vazamento",
                "modelo": melhor_sem_vazamento.get("modelo"),
                "accuracy": melhor_sem_vazamento.get("accuracy"),
                "balanced_accuracy": melhor_sem_vazamento.get("balanced_accuracy"),
                "f1_macro": melhor_sem_vazamento.get("f1_macro"),
                "f1_weighted": melhor_sem_vazamento.get("f1_weighted"),
                "observacao": "Scaler, imputador e KMeans ajustados apenas no treino ate 2022.",
            }
        )

    return pd.DataFrame(registros)


def main() -> None:
    garantir_diretorios()

    caminho_entrada = PROCESSED_DIR / "02_dataset_atributos.csv"
    verificar_arquivo(caminho_entrada)

    df = pd.read_csv(caminho_entrada)
    df = df[df["year"] >= ANO_MINIMO_ANALISE].copy()
    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    treino = df[df["year"] <= ANO_LIMITE_TREINO].copy()
    teste = df[df["year"] >= ANO_INICIO_TESTE].copy()

    if treino.empty or teste.empty:
        raise ValueError(
            "Validacao sem vazamento exige treino ate 2022 e teste de 2023 em diante."
        )

    variaveis = selecionar_variaveis_treino(treino)
    if len(variaveis) < 4:
        raise ValueError("Poucas variaveis disponiveis no treino para validacao sem vazamento.")

    X_train = treino[variaveis].apply(pd.to_numeric, errors="coerce")
    X_test = teste[variaveis].apply(pd.to_numeric, errors="coerce")

    imputador = SimpleImputer(strategy="median")
    escalador = StandardScaler()

    X_train_imp = imputador.fit_transform(X_train)
    X_test_imp = imputador.transform(X_test)
    X_train_proc = escalador.fit_transform(X_train_imp)
    X_test_proc = escalador.transform(X_test_imp)

    clusterizador = KMeans(n_clusters=K_CLUSTERS, random_state=RANDOM_STATE, n_init=50)
    labels_treino = clusterizador.fit_predict(X_train_proc)
    labels_teste = clusterizador.predict(X_test_proc)
    distancias_teste = clusterizador.transform(X_test_proc)

    mapa_nomes = nomear_clusters_por_treino(treino, labels_treino)
    y_train = pd.Series(labels_treino, index=treino.index).map(mapa_nomes)
    y_test = pd.Series(labels_teste, index=teste.index).map(mapa_nomes)

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

    modelos = carregar_modelos(len(treino))
    registros_metricas = []
    registros_predicoes = []
    estrategia = f"sem_vazamento_treino_ate_{ANO_LIMITE_TREINO}_teste_{ANO_INICIO_TESTE}_{int(teste['year'].max())}"

    for nome, modelo in modelos.items():
        print(f"Treinando validacao sem vazamento: {nome}")
        pipeline = Pipeline(
            steps=[
                ("preprocessador", preprocessador),
                ("modelo", modelo),
            ]
        )

        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            metricas = calcular_metricas_classificacao(y_test, y_pred, nome)
            metricas.update(
                {
                    "estrategia_validacao": estrategia,
                    "qtd_treino": len(treino),
                    "qtd_teste": len(teste),
                    "ano_limite_treino": ANO_LIMITE_TREINO,
                    "ano_inicio_teste": ANO_INICIO_TESTE,
                    "k_clusters_treino": K_CLUSTERS,
                    "qtd_variaveis": len(variaveis),
                }
            )
            registros_metricas.append(metricas)

            registros_predicoes.append(
                pd.DataFrame(
                    {
                        "country": teste["country"].values,
                        "year": teste["year"].values,
                        "cluster_real_sem_vazamento": labels_teste,
                        "perfil_real_sem_vazamento": y_test.values,
                        "perfil_predito": y_pred,
                        "modelo": nome,
                        "distancia_ao_centroide_real": distancias_teste[
                            np.arange(len(labels_teste)), labels_teste
                        ],
                        "estrategia_validacao": estrategia,
                    }
                )
            )
        except Exception as erro:
            print(f"Modelo {nome} falhou na validacao sem vazamento: {erro}")

    metricas_df = pd.DataFrame(registros_metricas).sort_values(
        ["f1_macro", "balanced_accuracy"],
        ascending=[False, False],
    )

    if metricas_df.empty:
        raise ValueError("Nenhum modelo foi treinado com sucesso na validacao sem vazamento.")

    predicoes_df = pd.concat(registros_predicoes, ignore_index=True)
    comparacao_df = comparar_com_validacao_original(metricas_df)

    salvar_csv(metricas_df, TABLES_DIR / "metricas_classificacao_sem_vazamento.csv")
    salvar_csv(predicoes_df, TABLES_DIR / "predicoes_classificacao_sem_vazamento.csv")
    salvar_csv(comparacao_df, TABLES_DIR / "comparacao_modelos_com_sem_vazamento.csv")

    print("\nValidacao temporal sem vazamento concluida.")
    print(f"Variaveis usadas: {variaveis}")
    print(metricas_df.to_string(index=False))


if __name__ == "__main__":
    main()
