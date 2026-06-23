# -*- coding: utf-8 -*-

"""
Script 05 - Classificação supervisionada dos perfis de transição.

Objetivo:
Treinar modelos supervisionados para reproduzir os perfis definidos pela
clusterização, usando validação temporal.

Execução:
python scripts/05_classificacao_ml.py
"""

from pathlib import Path
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.config import (  # noqa: E402
    PROCESSED_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    VARIAVEIS_CLUSTER_PREFERENCIAIS,
    RANDOM_STATE,
)
from utils.io_utils import garantir_diretorios, salvar_csv, verificar_arquivo  # noqa: E402
from utils.ml_utils import calcular_metricas_classificacao, matriz_confusao_para_dataframe  # noqa: E402

warnings.filterwarnings("ignore")


def carregar_modelos() -> dict:
    """Define modelos obrigatorios e adiciona opcionais compativeis quando possivel."""
    modelos = {
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5),
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svm_rbf": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            max_depth=None,
        ),
    }

    try:
        import xgboost  # noqa: F401

        print(
            "XGBoost instalado, mas ignorado nesta etapa para preservar "
            "rotulos textuais compativeis com a interpretabilidade."
        )
    except Exception:
        print("XGBoost nao instalado. Modelo ignorado.")

    try:
        from lightgbm import LGBMClassifier

        modelos["lightgbm"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            verbose=-1,
        )
    except Exception:
        print("LightGBM não instalado. Modelo ignorado.")

    return modelos


def definir_split_temporal(df: pd.DataFrame):
    """Cria divisão temporal. Se não houver anos recentes suficientes, usa fallback estratificado."""
    ano_max = int(df["year"].max())
    ano_teste_inicio = ano_max - 2

    treino = df[df["year"] < ano_teste_inicio].copy()
    teste = df[df["year"] >= ano_teste_inicio].copy()

    if treino["perfil_transicao"].nunique() < 2 or teste["perfil_transicao"].nunique() < 2 or len(teste) < 10:
        treino, teste = train_test_split(
            df,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=df["perfil_transicao"] if df["perfil_transicao"].nunique() > 1 else None,
        )
        estrategia = "estratificada_fallback"
    else:
        estrategia = f"temporal_treino_ate_{ano_teste_inicio - 1}_teste_{ano_teste_inicio}_{ano_max}"

    return treino, teste, estrategia


def main() -> None:
    garantir_diretorios()

    caminho_entrada = PROCESSED_DIR / "03_dataset_com_clusters.csv"
    verificar_arquivo(caminho_entrada)

    df = pd.read_csv(caminho_entrada)
    df = df.dropna(subset=["perfil_transicao"]).copy()

    variaveis = [v for v in VARIAVEIS_CLUSTER_PREFERENCIAIS if v in df.columns]
    variaveis = [v for v in variaveis if pd.to_numeric(df[v], errors="coerce").notna().mean() >= 0.5]

    if len(variaveis) < 4:
        raise ValueError("Poucas variáveis disponíveis para classificação.")

    treino, teste, estrategia = definir_split_temporal(df)

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

    modelos = carregar_modelos()
    registros_metricas = []
    predicoes = []
    modelos_treinados = {}

    for nome, modelo in modelos.items():
        print(f"Treinando modelo: {nome}")
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
            metricas["estrategia_validacao"] = estrategia
            metricas["qtd_treino"] = len(treino)
            metricas["qtd_teste"] = len(teste)
            registros_metricas.append(metricas)
            modelos_treinados[nome] = pipeline

            predicoes.append(
                pd.DataFrame(
                    {
                        "country": teste["country"].values,
                        "year": teste["year"].values,
                        "real": y_test.values,
                        "predito": y_pred,
                        "modelo": nome,
                    }
                )
            )
        except Exception as erro:
            print(f"Modelo {nome} falhou: {erro}")

    metricas_df = pd.DataFrame(registros_metricas).sort_values(
        ["f1_macro", "balanced_accuracy"], ascending=[False, False]
    )

    if metricas_df.empty:
        raise ValueError("Nenhum modelo supervisionado foi treinado com sucesso.")

    melhor_modelo_nome = metricas_df.iloc[0]["modelo"]
    melhor_pipeline = modelos_treinados[melhor_modelo_nome]
    y_pred_melhor = melhor_pipeline.predict(X_test)

    classes = sorted(y_test.unique())
    matriz = confusion_matrix(y_test, y_pred_melhor, labels=classes)
    matriz_df = matriz_confusao_para_dataframe(matriz, classes)

    relatorio = classification_report(y_test, y_pred_melhor, zero_division=0)
    (TABLES_DIR / "classification_report_melhor_modelo.txt").write_text(relatorio, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matriz)
    ax.set_title(f"Matriz de confusão - {melhor_modelo_nome}")
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe real")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matriz[i, j], ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "matriz_confusao_melhor_modelo.png", dpi=300)
    plt.close(fig)

    salvar_csv(metricas_df, TABLES_DIR / "metricas_classificacao.csv")
    salvar_csv(pd.concat(predicoes, ignore_index=True), TABLES_DIR / "predicoes_classificacao.csv")
    salvar_csv(matriz_df, TABLES_DIR / "matriz_confusao_melhor_modelo.csv", index=True)
    salvar_csv(pd.DataFrame({"variavel_usada": variaveis}), TABLES_DIR / "variaveis_usadas_classificacao.csv")

    joblib.dump(
        {
            "modelo": melhor_pipeline,
            "nome_modelo": melhor_modelo_nome,
            "variaveis": variaveis,
            "estrategia_validacao": estrategia,
        },
        MODELS_DIR / "melhor_modelo_classificacao.pkl",
    )

    print("\nClassificação concluída.")
    print(f"Melhor modelo: {melhor_modelo_nome}")
    print(metricas_df.to_string(index=False))


if __name__ == "__main__":
    main()
