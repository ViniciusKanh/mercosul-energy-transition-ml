# -*- coding: utf-8 -*-

"""
Script 02 - Auditoria dos dados.

Objetivo:
Verificar cobertura temporal, disponibilidade de países, variáveis candidatas,
valores ausentes e consistência inicial da base OWID.

Execução:
python scripts/02_auditoria_dados.py
"""

from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.config import (  # noqa: E402
    RAW_DIR,
    PROCESSED_DIR,
    TABLES_DIR,
    PAISES_MERCOSUL_ASSOCIADOS,
    VARIAVEIS_CANDIDATAS,
)
from utils.io_utils import garantir_diretorios, salvar_csv, verificar_arquivo  # noqa: E402


def main() -> None:
    garantir_diretorios()

    caminho_base = RAW_DIR / "owid-energy-data.csv"
    caminho_codebook = RAW_DIR / "owid-energy-codebook.csv"

    verificar_arquivo(caminho_base)
    verificar_arquivo(caminho_codebook)

    df = pd.read_csv(caminho_base)
    codebook = pd.read_csv(caminho_codebook)

    print(f"Base original: {df.shape[0]} linhas e {df.shape[1]} colunas.")

    colunas_obrigatorias = {"country", "year"}
    if not colunas_obrigatorias.issubset(df.columns):
        raise ValueError("A base não contém as colunas obrigatórias 'country' e 'year'.")

    df_mercosul = df[df["country"].isin(PAISES_MERCOSUL_ASSOCIADOS)].copy()

    if df_mercosul.empty:
        raise ValueError("Nenhum país do recorte MERCOSUL/associados foi encontrado na base.")

    cobertura = (
        df_mercosul.groupby("country")
        .agg(
            ano_minimo=("year", "min"),
            ano_maximo=("year", "max"),
            qtd_anos=("year", "nunique"),
            qtd_linhas=("year", "size"),
        )
        .reset_index()
        .sort_values("country")
    )
    cobertura["pais_previsto_no_projeto"] = cobertura["country"].isin(PAISES_MERCOSUL_ASSOCIADOS)

    paises_encontrados = set(df_mercosul["country"].unique())
    paises_ausentes = sorted(set(PAISES_MERCOSUL_ASSOCIADOS) - paises_encontrados)
    paises_ausentes_df = pd.DataFrame({"pais_ausente": paises_ausentes})

    registros_variaveis = []

    for coluna in VARIAVEIS_CANDIDATAS:
        existe = coluna in df_mercosul.columns
        registro = {
            "variavel": coluna,
            "existe_na_base": existe,
            "qtd_ausentes": None,
            "pct_ausentes": None,
            "minimo": None,
            "maximo": None,
        }

        if existe:
            serie = pd.to_numeric(df_mercosul[coluna], errors="coerce")
            registro["qtd_ausentes"] = int(serie.isna().sum())
            registro["pct_ausentes"] = round(float(serie.isna().mean() * 100), 2)
            registro["minimo"] = serie.min(skipna=True)
            registro["maximo"] = serie.max(skipna=True)

        registros_variaveis.append(registro)

    auditoria_variaveis = pd.DataFrame(registros_variaveis)

    variaveis_existentes = [v for v in VARIAVEIS_CANDIDATAS if v in df_mercosul.columns]
    registros_ausentes = []

    for pais, grupo in df_mercosul.groupby("country"):
        for coluna in variaveis_existentes:
            serie = pd.to_numeric(grupo[coluna], errors="coerce")
            registros_ausentes.append(
                {
                    "country": pais,
                    "variavel": coluna,
                    "qtd_ausentes": int(serie.isna().sum()),
                    "pct_ausentes": round(float(serie.isna().mean() * 100), 2),
                    "qtd_observacoes": int(len(grupo)),
                }
            )

    ausentes_por_pais = pd.DataFrame(registros_ausentes)

    codebook_filtrado = codebook.copy()
    if "column" in codebook.columns:
        codebook_filtrado = codebook[codebook["column"].isin(variaveis_existentes)].copy()

    salvar_csv(df_mercosul, PROCESSED_DIR / "01_mercosul_owid_filtrado.csv")
    salvar_csv(cobertura, TABLES_DIR / "auditoria_cobertura_paises.csv")
    salvar_csv(paises_ausentes_df, TABLES_DIR / "auditoria_paises_ausentes.csv")
    salvar_csv(auditoria_variaveis, TABLES_DIR / "auditoria_variaveis_candidatas.csv")
    salvar_csv(ausentes_por_pais, TABLES_DIR / "auditoria_valores_ausentes_por_pais.csv")
    salvar_csv(codebook_filtrado, TABLES_DIR / "dicionario_variaveis_candidatas.csv")

    print("\nAuditoria concluída.")
    print("Países encontrados:")
    print(sorted(df_mercosul["country"].unique()))
    print("\nCobertura temporal:")
    print(cobertura.to_string(index=False))


if __name__ == "__main__":
    main()
