# -*- coding: utf-8 -*-

"""
Funções auxiliares para avaliação de modelos.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def calcular_metricas_classificacao(y_true, y_pred, nome_modelo: str) -> dict:
    """Calcula métricas principais para classificação multiclasse."""
    return {
        "modelo": nome_modelo,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def matriz_confusao_para_dataframe(matriz, classes) -> pd.DataFrame:
    """Converte matriz de confusão em DataFrame identificado."""
    return pd.DataFrame(
        matriz,
        index=[f"real_{classe}" for classe in classes],
        columns=[f"pred_{classe}" for classe in classes],
    )
