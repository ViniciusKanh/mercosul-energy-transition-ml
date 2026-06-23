# -*- coding: utf-8 -*-

"""
Script 06 - Interpretabilidade do melhor modelo.

Objetivo:
Calcular importância por permutação e gerar uma figura interpretável das
variáveis mais relevantes para a classificação dos perfis.

Execução:
python scripts/06_interpretabilidade.py
"""

from pathlib import Path
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.config import PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR, RANDOM_STATE  # noqa: E402
from utils.io_utils import garantir_diretorios, salvar_csv, verificar_arquivo  # noqa: E402


def definir_split_para_interpretabilidade(df: pd.DataFrame):
    """Repete a lógica temporal usada na classificação."""
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

    return treino, teste


def main() -> None:
    garantir_diretorios()

    caminho_modelo = MODELS_DIR / "melhor_modelo_classificacao.pkl"
    caminho_dataset = PROCESSED_DIR / "03_dataset_com_clusters.csv"

    verificar_arquivo(caminho_modelo)
    verificar_arquivo(caminho_dataset)

    artefato = joblib.load(caminho_modelo)
    modelo = artefato["modelo"]
    nome_modelo = artefato["nome_modelo"]
    variaveis = artefato["variaveis"]

    df = pd.read_csv(caminho_dataset).dropna(subset=["perfil_transicao"]).copy()
    _, teste = definir_split_para_interpretabilidade(df)

    X_test = teste[variaveis].apply(pd.to_numeric, errors="coerce")
    y_test = teste["perfil_transicao"]

    resultado = permutation_importance(
        modelo,
        X_test,
        y_test,
        n_repeats=30,
        random_state=RANDOM_STATE,
        scoring="f1_macro",
    )

    importancia = pd.DataFrame(
        {
            "variavel": variaveis,
            "importancia_media": resultado.importances_mean,
            "importancia_desvio": resultado.importances_std,
            "modelo": nome_modelo,
        }
    ).sort_values("importancia_media", ascending=False)

    salvar_csv(importancia, TABLES_DIR / "importancia_permutacao.csv")

    top = importancia.head(15).sort_values("importancia_media", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["variavel"], top["importancia_media"], xerr=top["importancia_desvio"])
    ax.set_title(f"Importância por permutação - {nome_modelo}")
    ax.set_xlabel("Redução média no F1 macro")
    ax.set_ylabel("Variável")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "importancia_permutacao_top15.png", dpi=300)
    plt.close(fig)

    print("\nInterpretabilidade concluída.")
    print(importancia.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
