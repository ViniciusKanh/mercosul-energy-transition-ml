# -*- coding: utf-8 -*-

"""
Script 03 - Engenharia de atributos.

Objetivo:
Construir variaveis derivadas interpretaveis para analise de trajetorias de
transicao energetica e descarbonizacao.

Execucao:
python scripts/03_engenharia_atributos.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.config import PROCESSED_DIR, TABLES_DIR, VARIAVEIS_CLUSTER_PREFERENCIAIS  # noqa: E402
from utils.feature_utils import divisao_segura, normalizar_minmax, tendencia_movel_por_pais  # noqa: E402
from utils.io_utils import garantir_diretorios, salvar_csv, verificar_arquivo  # noqa: E402


def obter_coluna(df: pd.DataFrame, coluna: str) -> pd.Series:
    """Retorna coluna existente ou serie vazia com NaN."""
    if coluna in df.columns:
        return pd.to_numeric(df[coluna], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def adicionar_colunas_em_bloco(
    df: pd.DataFrame,
    novas_colunas: dict,
    sobrescrever: list[str] | None = None,
) -> pd.DataFrame:
    """Adiciona colunas em bloco para reduzir fragmentacao do DataFrame."""
    if not novas_colunas:
        return df

    sobrescrever = sobrescrever or []
    colunas_remover = [coluna for coluna in sobrescrever if coluna in df.columns]

    if colunas_remover:
        df = df.drop(columns=colunas_remover)

    bloco = pd.DataFrame(novas_colunas, index=df.index)
    df = pd.concat([df, bloco], axis=1)
    return df.copy()


def main() -> None:
    garantir_diretorios()

    caminho_entrada = PROCESSED_DIR / "01_mercosul_owid_filtrado.csv"
    verificar_arquivo(caminho_entrada)

    df = pd.read_csv(caminho_entrada)
    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    colunas_numericas = [coluna for coluna in df.columns if coluna not in ["country", "iso_code"]]
    df[colunas_numericas] = df[colunas_numericas].apply(pd.to_numeric, errors="coerce")
    df = df.copy()

    primary_energy = obter_coluna(df, "primary_energy_consumption")
    fossil_consumption = obter_coluna(df, "fossil_consumption")
    renewables_consumption = obter_coluna(df, "renewables_consumption")
    low_carbon_consumption = obter_coluna(df, "low_carbon_consumption")
    electricity_generation = obter_coluna(df, "electricity_generation")
    fossil_electricity = obter_coluna(df, "fossil_electricity")
    renewables_electricity = obter_coluna(df, "renewables_electricity")
    low_carbon_electricity = obter_coluna(df, "low_carbon_electricity")
    co2 = obter_coluna(df, "co2")

    colunas_base = {}

    if "fossil_share_energy" not in df.columns:
        colunas_base["fossil_share_energy"] = divisao_segura(fossil_consumption, primary_energy) * 100
    if "renewables_share_energy" not in df.columns:
        colunas_base["renewables_share_energy"] = divisao_segura(renewables_consumption, primary_energy) * 100
    if "low_carbon_share_energy" not in df.columns:
        colunas_base["low_carbon_share_energy"] = divisao_segura(low_carbon_consumption, primary_energy) * 100
    if "fossil_share_elec" not in df.columns:
        colunas_base["fossil_share_elec"] = divisao_segura(fossil_electricity, electricity_generation) * 100
    if "renewables_share_elec" not in df.columns:
        colunas_base["renewables_share_elec"] = divisao_segura(renewables_electricity, electricity_generation) * 100
    if "low_carbon_share_elec" not in df.columns:
        colunas_base["low_carbon_share_elec"] = divisao_segura(low_carbon_electricity, electricity_generation) * 100

    colunas_base["razao_renovavel_fossil_energy"] = divisao_segura(
        obter_coluna(df, "renewables_consumption"),
        obter_coluna(df, "fossil_consumption"),
    )

    if "co2_per_unit_energy" not in df.columns:
        colunas_base["co2_per_unit_energy"] = divisao_segura(co2, primary_energy)

    df = adicionar_colunas_em_bloco(
        df,
        colunas_base,
        sobrescrever=["razao_renovavel_fossil_energy"],
    )

    colunas_temporais = {}

    if "co2" in df.columns:
        colunas_temporais["variacao_co2_pct"] = (
            df.groupby("country")["co2"]
            .pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
            * 100
        )
    else:
        colunas_temporais["variacao_co2_pct"] = np.nan

    if "renewables_share_energy" in df.columns:
        colunas_temporais["variacao_renewables_share_energy_pp"] = (
            df.groupby("country")["renewables_share_energy"].diff()
        )
        colunas_temporais["tendencia_renovaveis_5anos"] = tendencia_movel_por_pais(
            df,
            coluna="renewables_share_energy",
            janela=5,
        )
    else:
        colunas_temporais["variacao_renewables_share_energy_pp"] = np.nan
        colunas_temporais["tendencia_renovaveis_5anos"] = np.nan

    if "co2" in df.columns:
        colunas_temporais["media_movel_co2_3anos"] = (
            df.groupby("country")["co2"]
            .rolling(window=3, min_periods=2)
            .mean()
            .reset_index(level=0, drop=True)
        )
    else:
        colunas_temporais["media_movel_co2_3anos"] = np.nan

    df = adicionar_colunas_em_bloco(
        df,
        colunas_temporais,
        sobrescrever=[
            "variacao_co2_pct",
            "variacao_renewables_share_energy_pp",
            "tendencia_renovaveis_5anos",
            "media_movel_co2_3anos",
        ],
    )

    componentes = []

    if "renewables_share_energy" in df.columns:
        componentes.append(normalizar_minmax(pd.to_numeric(df["renewables_share_energy"], errors="coerce")))
    if "low_carbon_share_elec" in df.columns:
        componentes.append(normalizar_minmax(pd.to_numeric(df["low_carbon_share_elec"], errors="coerce")))
    if "fossil_share_energy" in df.columns:
        componentes.append(1 - normalizar_minmax(pd.to_numeric(df["fossil_share_energy"], errors="coerce")))
    if "carbon_intensity_elec" in df.columns:
        componentes.append(1 - normalizar_minmax(pd.to_numeric(df["carbon_intensity_elec"], errors="coerce")))
    if "co2_per_unit_energy" in df.columns:
        componentes.append(1 - normalizar_minmax(pd.to_numeric(df["co2_per_unit_energy"], errors="coerce")))
    if "variacao_co2_pct" in df.columns:
        componentes.append(1 - normalizar_minmax(pd.to_numeric(df["variacao_co2_pct"], errors="coerce")))

    if componentes:
        matriz_componentes = pd.concat(componentes, axis=1)
        indice_transicao = matriz_componentes.mean(axis=1, skipna=True)
    else:
        indice_transicao = pd.Series(np.nan, index=df.index)

    df = adicionar_colunas_em_bloco(
        df,
        {"indice_transicao_preliminar": indice_transicao},
        sobrescrever=["indice_transicao_preliminar"],
    )

    variaveis_modelaveis = [v for v in VARIAVEIS_CLUSTER_PREFERENCIAIS if v in df.columns]
    auditoria_modelagem = []

    for coluna in variaveis_modelaveis:
        serie = pd.to_numeric(df[coluna], errors="coerce")
        auditoria_modelagem.append(
            {
                "variavel": coluna,
                "qtd_ausentes": int(serie.isna().sum()),
                "pct_ausentes": round(float(serie.isna().mean() * 100), 2),
                "minimo": serie.min(skipna=True),
                "maximo": serie.max(skipna=True),
                "media": serie.mean(skipna=True),
            }
        )

    salvar_csv(df, PROCESSED_DIR / "02_dataset_atributos.csv")
    salvar_csv(pd.DataFrame(auditoria_modelagem), TABLES_DIR / "auditoria_atributos_modelagem.csv")

    print("\nEngenharia de atributos concluida.")
    print(f"Variaveis modelaveis disponiveis: {len(variaveis_modelaveis)}")
    print(variaveis_modelaveis)


if __name__ == "__main__":
    main()
