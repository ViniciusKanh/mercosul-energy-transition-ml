# -*- coding: utf-8 -*-

"""
Funções auxiliares para engenharia de atributos energéticos e ambientais.
"""

import numpy as np
import pandas as pd


def divisao_segura(numerador, denominador):
    """Realiza divisão evitando infinito e divisão por zero."""
    resultado = np.where(
        (pd.notna(denominador)) & (denominador != 0),
        numerador / denominador,
        np.nan,
    )
    return resultado


def normalizar_minmax(serie: pd.Series) -> pd.Series:
    """Normaliza uma série para o intervalo [0, 1]."""
    minimo = serie.min(skipna=True)
    maximo = serie.max(skipna=True)

    if pd.isna(minimo) or pd.isna(maximo) or maximo == minimo:
        return pd.Series(np.nan, index=serie.index)

    return (serie - minimo) / (maximo - minimo)


def calcular_inclinacao(y: pd.Series) -> float:
    """
    Calcula a inclinação linear simples de uma sequência temporal.

    A função ignora valores ausentes. Retorna NaN se houver menos de 3 pontos válidos.
    """
    y = y.dropna()
    if len(y) < 3:
        return np.nan

    x = np.arange(len(y))
    coeficientes = np.polyfit(x, y.values, 1)
    return float(coeficientes[0])


def tendencia_movel_por_pais(
    df: pd.DataFrame,
    coluna: str,
    janela: int = 5,
    coluna_pais: str = "country",
) -> pd.Series:
    """Calcula tendência móvel por país usando inclinação linear em janela deslizante."""
    resultados = []

    for _, grupo in df.sort_values([coluna_pais, "year"]).groupby(coluna_pais):
        serie = grupo[coluna]
        tendencia = serie.rolling(window=janela, min_periods=3).apply(calcular_inclinacao, raw=False)
        resultados.append(tendencia)

    return pd.concat(resultados).sort_index()
