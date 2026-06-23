# -*- coding: utf-8 -*-

"""
Script 13 - Exportacao de tabelas para LaTeX.

Objetivo:
Gerar tabelas .tex compactas para uso posterior no artigo/Overleaf.

Execucao:
python scripts/13_exportar_tabelas_latex.py
"""

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.config import TABLES_DIR, RESULTS_DIR  # noqa: E402
from utils.io_utils import garantir_diretorios  # noqa: E402

LATEX_DIR = RESULTS_DIR / "latex"


def escapar_latex(valor: object) -> str:
    """Escapa caracteres especiais de LaTeX."""
    if pd.isna(valor):
        return ""

    if isinstance(valor, float):
        texto = f"{valor:.4f}".rstrip("0").rstrip(".")
    else:
        texto = str(valor)

    substituicoes = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }

    for original, novo in substituicoes.items():
        texto = texto.replace(original, novo)
    return texto


def dataframe_para_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Converte DataFrame em tabular LaTeX simples, sem dependencias extras."""
    if df is None or df.empty:
        return (
            "% Tabela nao gerada: arquivo ausente ou vazio.\n"
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            f"\\caption{{{escapar_latex(caption)}}}\n"
            f"\\label{{{escapar_latex(label)}}}\n"
            "\\begin{tabular}{l}\n"
            "\\hline\n"
            "Sem dados disponiveis \\\\\n"
            "\\hline\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )

    colunas = list(df.columns)
    alinhamento = "l" * len(colunas)
    linhas = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{escapar_latex(caption)}}}",
        f"\\label{{{escapar_latex(label)}}}",
        f"\\begin{{tabular}}{{{alinhamento}}}",
        "\\hline",
        " & ".join(escapar_latex(c) for c in colunas) + r" \\",
        "\\hline",
    ]

    for _, linha in df.iterrows():
        linhas.append(" & ".join(escapar_latex(linha[c]) for c in colunas) + r" \\")

    linhas.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(linhas)


def ler_tabela(nome_arquivo: str) -> pd.DataFrame:
    """Le uma tabela CSV opcional."""
    caminho = TABLES_DIR / nome_arquivo
    if not caminho.exists():
        print(f"Tabela nao encontrada para LaTeX: {caminho}")
        return pd.DataFrame()
    return pd.read_csv(caminho)


def selecionar_colunas(df: pd.DataFrame, colunas: list[str]) -> pd.DataFrame:
    """Seleciona colunas existentes, preservando a ordem desejada."""
    existentes = [c for c in colunas if c in df.columns]
    if not existentes:
        return df
    return df[existentes].copy()


def salvar_tabela(df: pd.DataFrame, nome_saida: str, caption: str, label: str) -> None:
    """Salva uma tabela LaTeX."""
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    caminho = LATEX_DIR / nome_saida
    caminho.write_text(dataframe_para_latex(df, caption, label), encoding="utf-8")
    print(f"Tabela LaTeX salva: {caminho}")


def main() -> None:
    garantir_diretorios()
    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    cobertura = selecionar_colunas(
        ler_tabela("auditoria_cobertura_paises.csv"),
        ["country", "ano_minimo", "ano_maximo", "qtd_anos", "qtd_linhas"],
    )
    salvar_tabela(
        cobertura,
        "tabela_cobertura_paises.tex",
        "Cobertura temporal dos paises analisados.",
        "tab:cobertura-paises",
    )

    metricas_cluster = selecionar_colunas(
        ler_tabela("metricas_clusterizacao.csv").head(12),
        ["modelo_cluster", "k", "silhouette", "davies_bouldin", "calinski_harabasz"],
    )
    salvar_tabela(
        metricas_cluster,
        "tabela_metricas_clusterizacao.tex",
        "Metricas internas dos modelos de clusterizacao.",
        "tab:metricas-clusterizacao",
    )

    metricas_classificacao = selecionar_colunas(
        ler_tabela("metricas_classificacao.csv"),
        ["modelo", "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "estrategia_validacao"],
    )
    salvar_tabela(
        metricas_classificacao,
        "tabela_metricas_classificacao.tex",
        "Metricas da classificacao supervisionada dos perfis.",
        "tab:metricas-classificacao",
    )

    importancia = selecionar_colunas(
        ler_tabela("importancia_permutacao.csv").head(15),
        ["variavel", "importancia_media", "importancia_desvio", "modelo"],
    )
    salvar_tabela(
        importancia,
        "tabela_importancia_variaveis.tex",
        "Importancia das variaveis por permutacao.",
        "tab:importancia-variaveis",
    )

    perfis = ler_tabela("perfis_clusters_interpretados.csv")
    if perfis.empty:
        perfis = ler_tabela("perfis_clusters.csv")
    perfis = selecionar_colunas(
        perfis,
        [
            "cluster_transicao",
            "nome_sugerido",
            "caracteristica_principal",
            "qtd_observacoes",
            "qtd_paises",
            "ano_minimo",
            "ano_maximo",
        ],
    )
    salvar_tabela(
        perfis,
        "tabela_perfis_clusters.tex",
        "Perfis interpretados dos clusters de transicao energetica.",
        "tab:perfis-clusters",
    )

    validacao = selecionar_colunas(
        ler_tabela("metricas_classificacao_sem_vazamento.csv"),
        ["modelo", "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "estrategia_validacao"],
    )
    salvar_tabela(
        validacao,
        "tabela_validacao_sem_vazamento.tex",
        "Validacao temporal sem vazamento metodologico.",
        "tab:validacao-sem-vazamento",
    )

    robustez = ler_tabela("robustez_clusterizacao.csv")
    if not robustez.empty:
        robustez = robustez.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True]).head(20)
    robustez = selecionar_colunas(
        robustez,
        ["algoritmo", "k", "seed", "silhouette", "davies_bouldin", "calinski_harabasz"],
    )
    salvar_tabela(
        robustez,
        "tabela_robustez_clusterizacao.tex",
        "Robustez da clusterizacao por algoritmo, k e seed.",
        "tab:robustez-clusterizacao",
    )

    print("\nExportacao LaTeX concluida.")


if __name__ == "__main__":
    main()
