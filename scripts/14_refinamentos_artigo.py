# -*- coding: utf-8 -*-

"""
Script 14 - Refinamentos finais para escrita do artigo.

Objetivo:
Gerar tabelas e figuras finais para uso no artigo sem alterar a metodologia
central, a escolha do KMeans k=3 ou os rotulos de cluster ja consolidados.

Execucao:
python scripts/14_refinamentos_artigo.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - fallback para ambiente minimo
    sns = None

from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.config import PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, RESULTS_DIR  # noqa: E402
from utils.io_utils import garantir_diretorios, salvar_csv, verificar_arquivo  # noqa: E402

LATEX_DIR = RESULTS_DIR / "latex"

ORDEM_PERFIS = [
    "Intermediário",
    "Fóssil-intensivo",
    "Renovável/baixo carbono",
]

CORES_PERFIS = ["#4C78A8", "#E45756", "#54A24B"]

VARIAVEIS_HEATMAP_ARTIGO = [
    "renewables_share_energy",
    "fossil_share_energy",
    "low_carbon_share_energy",
    "renewables_share_elec",
    "fossil_share_elec",
    "low_carbon_share_elec",
    "carbon_intensity_elec",
    "energy_per_capita",
    "energy_per_gdp",
    "variacao_renewables_share_energy_pp",
    "tendencia_renovaveis_5anos",
    "indice_transicao_preliminar",
]

NOMES_HEATMAP_ARTIGO = {
    "renewables_share_energy": "Renováveis energia",
    "fossil_share_energy": "Fósseis energia",
    "low_carbon_share_energy": "Baixo carbono energia",
    "renewables_share_elec": "Renováveis eletricidade",
    "fossil_share_elec": "Fósseis eletricidade",
    "low_carbon_share_elec": "Baixo carbono eletricidade",
    "carbon_intensity_elec": "Intensidade carbono",
    "energy_per_capita": "Energia per capita",
    "energy_per_gdp": "Energia por PIB",
    "variacao_renewables_share_energy_pp": "Variação renováveis",
    "tendencia_renovaveis_5anos": "Tendência renováveis",
    "indice_transicao_preliminar": "Índice transição",
}

VARIAVEIS_COMPLETAS_ANO_RECENTE = [
    "renewables_share_energy",
    "fossil_share_energy",
    "renewables_share_elec",
    "fossil_share_elec",
    "carbon_intensity_elec",
    "indice_transicao_preliminar",
]

COLUNAS_ANO_RECENTE_COMPLETO = [
    "country",
    "year",
    "perfil_transicao",
    "nome_sugerido",
    "renewables_share_energy",
    "fossil_share_energy",
    "renewables_share_elec",
    "fossil_share_elec",
    "carbon_intensity_elec",
    "indice_transicao_preliminar",
    "observacao_completude",
]

COLUNAS_CLUSTER_INTERMEDIARIO = [
    "country",
    "year",
    "perfil_transicao",
    "renewables_share_energy",
    "fossil_share_energy",
    "renewables_share_elec",
    "fossil_share_elec",
    "carbon_intensity_elec",
    "energy_per_capita",
    "energy_per_gdp",
    "indice_transicao_preliminar",
]

PAISES_CHAVE = [
    "Argentina",
    "Bolivia",
    "Brazil",
    "Chile",
    "Paraguay",
    "Uruguay",
    "Venezuela",
]


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega dataset clusterizado e nomes interpretaveis dos clusters."""
    caminho_dataset = PROCESSED_DIR / "03_dataset_com_clusters.csv"
    caminho_nomes = TABLES_DIR / "nomes_clusters_sugeridos.csv"
    verificar_arquivo(caminho_dataset)
    verificar_arquivo(caminho_nomes)

    return pd.read_csv(caminho_dataset), pd.read_csv(caminho_nomes)


def nome_curto_perfil(nome_sugerido: object, perfil_transicao: object = None) -> str:
    """Converte nomes longos dos perfis em rotulos curtos para artigo."""
    texto = f"{nome_sugerido or ''} {perfil_transicao or ''}".lower()

    if "intermedi" in texto:
        return "Intermediário"
    if "fossil" in texto or "fóssil" in texto or "baixa_transicao" in texto:
        return "Fóssil-intensivo"
    if "renov" in texto or "baixo carbono" in texto or "avancada" in texto:
        return "Renovável/baixo carbono"

    return str(nome_sugerido or perfil_transicao or "Perfil sem nome")


def preparar_dataset_com_nomes(df: pd.DataFrame, nomes: pd.DataFrame) -> pd.DataFrame:
    """Anexa nome sugerido e rotulo curto ao dataset clusterizado."""
    dados = df.merge(
        nomes[["cluster_transicao", "nome_sugerido"]],
        on="cluster_transicao",
        how="left",
    ).copy()

    perfil_artigo = dados.apply(
        lambda linha: nome_curto_perfil(linha.get("nome_sugerido"), linha.get("perfil_transicao")),
        axis=1,
    )
    return pd.concat([dados, perfil_artigo.rename("perfil_artigo")], axis=1).copy()


def salvar_figura(fig: plt.Figure, nome_arquivo: str) -> None:
    """Salva figura em PNG."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    caminho = FIGURES_DIR / nome_arquivo
    fig.tight_layout()
    fig.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva: {caminho}")


def escapar_latex(valor: object) -> str:
    """Escapa caracteres especiais de LaTeX."""
    if pd.isna(valor):
        return ""

    if isinstance(valor, float):
        texto = f"{valor:.3f}".rstrip("0").rstrip(".")
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


def salvar_latex(df: pd.DataFrame, caminho: Path, caption: str, label: str) -> None:
    """Salva DataFrame como tabela LaTeX simples."""
    caminho.parent.mkdir(parents=True, exist_ok=True)

    colunas = list(df.columns)
    linhas = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{escapar_latex(caption)}}}",
        f"\\label{{{escapar_latex(label)}}}",
        f"\\begin{{tabular}}{{{'l' * len(colunas)}}}",
        "\\hline",
        " & ".join(escapar_latex(coluna) for coluna in colunas) + r" \\",
        "\\hline",
    ]

    for _, linha in df.iterrows():
        linhas.append(" & ".join(escapar_latex(linha[coluna]) for coluna in colunas) + r" \\")

    linhas.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    caminho.write_text("\n".join(linhas), encoding="utf-8")
    print(f"Tabela LaTeX salva: {caminho}")


def gerar_tabela_paises_ano_recente_completa(dados: pd.DataFrame) -> None:
    """Seleciona, por pais, o ano mais recente com todos os indicadores completos."""
    faltantes = [coluna for coluna in VARIAVEIS_COMPLETAS_ANO_RECENTE if coluna not in dados.columns]
    if faltantes:
        raise ValueError(f"Colunas obrigatorias ausentes para ano recente completo: {faltantes}")

    registros = []
    for pais, grupo in dados.sort_values(["country", "year"]).groupby("country"):
        validos = grupo.dropna(subset=VARIAVEIS_COMPLETAS_ANO_RECENTE)
        if validos.empty:
            registro = {coluna: np.nan for coluna in COLUNAS_ANO_RECENTE_COMPLETO}
            registro["country"] = pais
            registro["observacao_completude"] = "Sem ano com todos os indicadores completos"
        else:
            registro = validos.tail(1).iloc[0][
                [coluna for coluna in COLUNAS_ANO_RECENTE_COMPLETO if coluna != "observacao_completude"]
            ].to_dict()
            registro["observacao_completude"] = "Completo"
        registros.append(registro)

    recentes = pd.DataFrame(registros)[COLUNAS_ANO_RECENTE_COMPLETO]
    recentes = recentes.sort_values("country").reset_index(drop=True)

    salvar_csv(recentes, TABLES_DIR / "paises_ano_recente_indicadores_completos.csv")
    salvar_latex(
        recentes,
        LATEX_DIR / "tabela_paises_ano_recente_indicadores_completos.tex",
        "Indicadores completos no ano mais recente disponivel por pais.",
        "tab:paises-ano-recente-indicadores-completos",
    )


def gerar_tabela_cluster_intermediario(dados: pd.DataFrame) -> None:
    """Lista todos os registros do cluster intermediario original (cluster 0)."""
    cluster_intermediario = (
        dados[dados["cluster_transicao"] == 0]
        .sort_values(["country", "year"])[COLUNAS_CLUSTER_INTERMEDIARIO]
        .reset_index(drop=True)
    )

    salvar_csv(cluster_intermediario, TABLES_DIR / "cluster_intermediario_detalhado.csv")
    salvar_latex(
        cluster_intermediario,
        LATEX_DIR / "tabela_cluster_intermediario_detalhado.tex",
        "Registros do cluster intermediario.",
        "tab:cluster-intermediario-detalhado",
    )


def gerar_heatmap_clusters_pais_ano(dados: pd.DataFrame) -> None:
    """Gera heatmap pais x ano com perfis interpretaveis."""
    mapa_perfis = {perfil: indice for indice, perfil in enumerate(ORDEM_PERFIS)}
    dados_heatmap = pd.DataFrame(
        {
            "country": dados["country"].values,
            "year": dados["year"].values,
            "codigo_perfil": dados["perfil_artigo"].map(mapa_perfis).values,
        }
    )
    matriz = (
        dados_heatmap
        .pivot_table(index="country", columns="year", values="codigo_perfil", aggfunc="first")
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(14, 6.5))
    cmap = ListedColormap(CORES_PERFIS)
    norm = BoundaryNorm(np.arange(-0.5, len(ORDEM_PERFIS) + 0.5, 1), cmap.N)
    im = ax.imshow(matriz.values, aspect="auto", cmap=cmap, norm=norm)

    anos = matriz.columns.tolist()
    passo = 2 if len(anos) <= 35 else 5
    ticks_x = np.arange(0, len(anos), passo)
    ax.set_xticks(ticks_x)
    ax.set_xticklabels([str(anos[i]) for i in ticks_x], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matriz.index)))
    ax.set_yticklabels(matriz.index)
    ax.set_xlabel("Ano")
    ax.set_ylabel("País")
    ax.set_title("Perfis de transição energética por país e ano")

    cbar = fig.colorbar(im, ax=ax, ticks=range(len(ORDEM_PERFIS)), fraction=0.025, pad=0.02)
    cbar.ax.set_yticklabels(ORDEM_PERFIS)
    cbar.set_label("Perfil de transição")
    salvar_figura(fig, "heatmap_clusters_pais_ano.png")


def gerar_heatmap_centroides_artigo(dados: pd.DataFrame) -> None:
    """Gera heatmap dos indicadores padronizados com rotulos fixos de artigo."""
    variaveis = [coluna for coluna in VARIAVEIS_HEATMAP_ARTIGO if coluna in dados.columns]
    if len(variaveis) != len(VARIAVEIS_HEATMAP_ARTIGO):
        ausentes = sorted(set(VARIAVEIS_HEATMAP_ARTIGO) - set(variaveis))
        raise ValueError(f"Variaveis ausentes no heatmap de centroides: {ausentes}")

    X = dados[variaveis].apply(pd.to_numeric, errors="coerce")
    preprocessador = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_proc = preprocessador.fit_transform(X)
    padronizado = pd.DataFrame(X_proc, columns=variaveis)
    padronizado["perfil_artigo"] = dados["perfil_artigo"].values

    medias = padronizado.groupby("perfil_artigo")[variaveis].mean()
    medias = medias.reindex(ORDEM_PERFIS)
    medias.columns = [NOMES_HEATMAP_ARTIGO[coluna] for coluna in variaveis]

    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    usou_seaborn = sns is not None
    if usou_seaborn:
        sns.heatmap(
            medias,
            cmap="RdBu_r",
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.4,
            cbar_kws={"label": "Média padronizada"},
            ax=ax,
        )
    else:
        im = ax.imshow(medias.values, cmap="RdBu_r", aspect="auto", vmin=-1.3, vmax=1.3)
        ax.set_xticks(np.arange(medias.shape[1]))
        ax.set_yticks(np.arange(medias.shape[0]))
        fig.colorbar(im, ax=ax, label="Média padronizada")

    ax.set_title("Médias padronizadas dos indicadores por perfil de transição energética")
    ax.set_xlabel("Indicador")
    ax.set_ylabel("Perfil de transição")
    deslocamento = 0.5 if usou_seaborn else 0
    ax.set_xticks(np.arange(medias.shape[1]) + deslocamento)
    ax.set_yticks(np.arange(medias.shape[0]) + deslocamento)
    ax.set_xticklabels(medias.columns, rotation=45, ha="right")
    ax.set_yticklabels(medias.index, rotation=0)
    salvar_figura(fig, "heatmap_centroides_artigo.png")


def mapa_perfis_para_artigo(dados: pd.DataFrame) -> dict[str, str]:
    """Cria mapa entre perfil_transicao original e nome curto do artigo."""
    return (
        dados[["perfil_transicao", "perfil_artigo"]]
        .dropna()
        .drop_duplicates()
        .set_index("perfil_transicao")["perfil_artigo"]
        .to_dict()
    )


def gerar_matriz_confusao_artigo(dados: pd.DataFrame) -> None:
    """Gera matriz de confusao da validacao temporal sem vazamento."""
    caminho_metricas = TABLES_DIR / "metricas_classificacao_sem_vazamento.csv"
    caminho_predicoes = TABLES_DIR / "predicoes_classificacao_sem_vazamento.csv"
    verificar_arquivo(caminho_metricas)
    verificar_arquivo(caminho_predicoes)

    metricas = pd.read_csv(caminho_metricas)
    predicoes = pd.read_csv(caminho_predicoes)
    melhor_modelo = (
        metricas.sort_values(["f1_macro", "balanced_accuracy"], ascending=[False, False])
        .iloc[0]["modelo"]
    )

    predicoes_modelo = predicoes[predicoes["modelo"] == melhor_modelo].copy()
    mapa_classes = mapa_perfis_para_artigo(dados)
    predicoes_modelo["real_legivel"] = (
        predicoes_modelo["perfil_real_sem_vazamento"].map(mapa_classes).fillna(predicoes_modelo["perfil_real_sem_vazamento"])
    )
    predicoes_modelo["predito_legivel"] = (
        predicoes_modelo["perfil_predito"].map(mapa_classes).fillna(predicoes_modelo["perfil_predito"])
    )

    y_true = predicoes_modelo["real_legivel"]
    y_pred = predicoes_modelo["predito_legivel"]
    matriz = confusion_matrix(y_true, y_pred, labels=ORDEM_PERFIS)
    matriz_df = pd.DataFrame(matriz, index=ORDEM_PERFIS, columns=ORDEM_PERFIS)
    salvar_csv(matriz_df, TABLES_DIR / "matriz_confusao_artigo.csv", index=True)

    diagnostico = []
    for classe in ORDEM_PERFIS:
        qtd_real = int((y_true == classe).sum())
        qtd_predito = int((y_pred == classe).sum())
        diagnostico.append(
            {
                "modelo": melhor_modelo,
                "classe": classe,
                "presente_no_teste": qtd_real > 0,
                "qtd_real_teste": qtd_real,
                "qtd_predito": qtd_predito,
                "linha_zerada": qtd_real == 0,
                "coluna_zerada": qtd_predito == 0,
                "observacao": (
                    "Perfil ausente no conjunto de teste temporal; interpretar a linha/coluna zerada como limitação da amostra."
                    if qtd_real == 0 or qtd_predito == 0
                    else "Perfil presente no conjunto de teste temporal."
                ),
            }
        )

    salvar_csv(pd.DataFrame(diagnostico), TABLES_DIR / "diagnostico_matriz_confusao_artigo.csv")

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    usou_seaborn = sns is not None
    if usou_seaborn:
        sns.heatmap(
            matriz_df,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            linewidths=0.5,
            ax=ax,
        )
    else:
        im = ax.imshow(matriz_df.values, cmap="Blues")
        ax.set_xticks(np.arange(matriz_df.shape[1]))
        ax.set_yticks(np.arange(matriz_df.shape[0]))
        for i in range(matriz_df.shape[0]):
            for j in range(matriz_df.shape[1]):
                ax.text(j, i, str(matriz_df.iloc[i, j]), ha="center", va="center")
        fig.colorbar(im, ax=ax)

    ax.set_title("Matriz de confusão da classificação temporal sem vazamento")
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe real")
    deslocamento = 0.5 if usou_seaborn else 0
    ax.set_xticks(np.arange(len(ORDEM_PERFIS)) + deslocamento)
    ax.set_yticks(np.arange(len(ORDEM_PERFIS)) + deslocamento)
    ax.set_xticklabels(ORDEM_PERFIS, rotation=35, ha="right")
    ax.set_yticklabels(ORDEM_PERFIS, rotation=0)
    salvar_figura(fig, "matriz_confusao_artigo.png")


def gerar_serie_temporal_paises_chave(
    dados: pd.DataFrame,
    coluna: str,
    titulo: str,
    ylabel: str,
    nome_arquivo: str,
) -> None:
    """Gera serie temporal filtrada para paises-chave do artigo."""
    dados_filtrados = dados[dados["country"].isin(PAISES_CHAVE)].copy()
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for pais, grupo in dados_filtrados.sort_values("year").groupby("country"):
        ax.plot(
            grupo["year"],
            pd.to_numeric(grupo[coluna], errors="coerce"),
            marker="o",
            markersize=2.6,
            linewidth=1.4,
            label=pais,
        )

    ax.set_title(titulo)
    ax.set_xlabel("Ano")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    salvar_figura(fig, nome_arquivo)


def gerar_series_temporais_filtradas(dados: pd.DataFrame) -> None:
    """Gera as duas series filtradas solicitadas para paises-chave."""
    gerar_serie_temporal_paises_chave(
        dados,
        "carbon_intensity_elec",
        "Evolução da intensidade de carbono da eletricidade em países-chave",
        "Intensidade de carbono da eletricidade",
        "evolucao_intensidade_carbono_paises_chave.png",
    )
    gerar_serie_temporal_paises_chave(
        dados,
        "renewables_share_energy",
        "Evolução da participação de renováveis na energia em países-chave",
        "Renováveis na energia (%)",
        "evolucao_renovaveis_paises_chave.png",
    )


def main() -> None:
    garantir_diretorios()
    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    df, nomes = carregar_dados()
    dados = preparar_dataset_com_nomes(df, nomes)

    gerar_tabela_paises_ano_recente_completa(dados)
    gerar_tabela_cluster_intermediario(dados)
    gerar_heatmap_clusters_pais_ano(dados)
    gerar_heatmap_centroides_artigo(dados)
    gerar_matriz_confusao_artigo(dados)
    gerar_series_temporais_filtradas(dados)

    print("\nRefinamentos finais para artigo concluidos.")


if __name__ == "__main__":
    main()
