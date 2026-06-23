# -*- coding: utf-8 -*-

"""
Script 11 - Visualizacoes cientificas dos resultados.

Objetivo:
Gerar figuras de apoio ao artigo a partir dos resultados consolidados do
pipeline experimental.

Execucao:
python scripts/11_visualizacoes_resultados.py
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - fallback para ambiente minimo
    sns = None

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.config import PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, VARIAVEIS_CLUSTER_PREFERENCIAIS, RANDOM_STATE  # noqa: E402
from utils.io_utils import garantir_diretorios, verificar_arquivo  # noqa: E402

warnings.filterwarnings("ignore")


def salvar_figura(fig: plt.Figure, nome_arquivo: str) -> None:
    """Salva figura em PNG com resolucao adequada para relatorio/artigo."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    caminho = FIGURES_DIR / nome_arquivo
    fig.tight_layout()
    fig.savefig(caminho, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva: {caminho}")


def carregar_variaveis(df: pd.DataFrame) -> list[str]:
    """Usa a lista de variaveis da clusterizacao quando disponivel."""
    caminho_variaveis = TABLES_DIR / "variaveis_usadas_clusterizacao.csv"
    if caminho_variaveis.exists():
        variaveis = pd.read_csv(caminho_variaveis)["variavel_usada"].dropna().tolist()
    else:
        variaveis = [v for v in VARIAVEIS_CLUSTER_PREFERENCIAIS if v in df.columns]

    return [v for v in variaveis if v in df.columns]


def matriz_preprocessada(df: pd.DataFrame, variaveis: list[str]) -> np.ndarray:
    """Imputa e padroniza matriz numerica para PCA, UMAP e heatmap."""
    X = df[variaveis].apply(pd.to_numeric, errors="coerce")
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return pipeline.fit_transform(X)


def plot_heatmap(dados: pd.DataFrame, titulo: str, caminho: str, figsize: tuple[int, int] = (12, 6)) -> None:
    """Gera heatmap usando seaborn quando disponivel, com fallback em matplotlib."""
    fig, ax = plt.subplots(figsize=figsize)

    if sns is not None:
        sns.heatmap(dados, cmap="RdBu_r", center=0, annot=True, fmt=".2f", linewidths=0.4, ax=ax)
    else:
        im = ax.imshow(dados.values, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(np.arange(len(dados.columns)))
        ax.set_xticklabels(dados.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(dados.index)))
        ax.set_yticklabels(dados.index)
        fig.colorbar(im, ax=ax)

    ax.set_title(titulo)
    ax.set_xlabel("Variavel")
    ax.set_ylabel("Cluster")
    salvar_figura(fig, caminho)


def gerar_pca(df: pd.DataFrame, variaveis: list[str], X_proc: np.ndarray) -> None:
    """Gera grafico PCA 2D dos clusters."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_proc)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=df["cluster_transicao"],
        cmap="viridis",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_title("PCA 2D dos perfis de transicao energetica")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% da variancia)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% da variancia)")
    ax.grid(True, alpha=0.25)
    legenda = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    ax.add_artist(legenda)
    salvar_figura(fig, "pca_clusters.png")


def gerar_umap_opcional(df: pd.DataFrame, X_proc: np.ndarray) -> None:
    """Gera UMAP 2D se umap-learn estiver instalado."""
    try:
        import umap
    except ImportError:
        print("umap-learn nao instalado. Figura UMAP ignorada.")
        return

    redutor = umap.UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1)
    coords = redutor.fit_transform(X_proc)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=df["cluster_transicao"],
        cmap="viridis",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_title("UMAP 2D dos perfis de transicao energetica")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.25)
    legenda = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    ax.add_artist(legenda)
    salvar_figura(fig, "umap_clusters.png")


def gerar_heatmap_centroides(df: pd.DataFrame, variaveis: list[str], X_proc: np.ndarray) -> None:
    """Gera heatmap de medias padronizadas por cluster."""
    padronizado = pd.DataFrame(X_proc, columns=variaveis)
    padronizado["cluster_transicao"] = df["cluster_transicao"].values
    medias = padronizado.groupby("cluster_transicao")[variaveis].mean()
    plot_heatmap(medias, "Medias padronizadas por cluster", "heatmap_centroides.png")


def gerar_evolucao_clusters(df: pd.DataFrame) -> None:
    """Gera serie temporal dos clusters por pais."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for pais, grupo in df.sort_values("year").groupby("country"):
        ax.plot(grupo["year"], grupo["cluster_transicao"], marker="o", linewidth=1.2, markersize=3, label=pais)

    ax.set_title("Evolucao temporal dos clusters por pais")
    ax.set_xlabel("Ano")
    ax.set_ylabel("Cluster de transicao")
    ax.set_yticks(sorted(df["cluster_transicao"].dropna().unique()))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    salvar_figura(fig, "evolucao_clusters_pais.png")


def gerar_evolucao_variavel(df: pd.DataFrame, coluna: str, titulo: str, ylabel: str, arquivo: str) -> None:
    """Gera serie temporal de uma variavel por pais."""
    if coluna not in df.columns:
        print(f"Coluna ausente para visualizacao: {coluna}")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    for pais, grupo in df.sort_values("year").groupby("country"):
        serie = pd.to_numeric(grupo[coluna], errors="coerce")
        ax.plot(grupo["year"], serie, linewidth=1.4, label=pais)

    ax.set_title(titulo)
    ax.set_xlabel("Ano")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    salvar_figura(fig, arquivo)


def gerar_importancia_permutacao() -> None:
    """Gera grafico de importancia por permutacao a partir da tabela existente."""
    caminho = TABLES_DIR / "importancia_permutacao.csv"
    if not caminho.exists():
        print("Tabela de importancia por permutacao nao encontrada. Figura ignorada.")
        return

    importancia = pd.read_csv(caminho)
    if importancia.empty:
        print("Tabela de importancia por permutacao vazia. Figura ignorada.")
        return

    top = importancia.sort_values("importancia_media", ascending=False).head(15)
    top = top.sort_values("importancia_media", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["variavel"], top["importancia_media"], xerr=top.get("importancia_desvio"))
    ax.set_title("Importancia das variaveis por permutacao")
    ax.set_xlabel("Reducao media no F1 macro")
    ax.set_ylabel("Variavel")
    ax.grid(True, axis="x", alpha=0.25)
    salvar_figura(fig, "importancia_permutacao.png")


def gerar_matriz_confusao() -> None:
    """Gera heatmap da matriz de confusao do melhor modelo."""
    caminho = TABLES_DIR / "matriz_confusao_melhor_modelo.csv"
    if not caminho.exists():
        print("Tabela da matriz de confusao nao encontrada. Figura ignorada.")
        return

    matriz = pd.read_csv(caminho)
    if matriz.columns[0].startswith("Unnamed") or matriz.columns[0].startswith("real_"):
        matriz = pd.read_csv(caminho, index_col=0)

    matriz = matriz.apply(pd.to_numeric, errors="coerce").fillna(0)
    plot_heatmap(matriz, "Matriz de confusao do melhor modelo", "matriz_confusao_melhor_modelo.png", figsize=(8, 6))


def main() -> None:
    garantir_diretorios()

    caminho_dataset = PROCESSED_DIR / "03_dataset_com_clusters.csv"
    verificar_arquivo(caminho_dataset)

    df = pd.read_csv(caminho_dataset)
    df = df.dropna(subset=["cluster_transicao"]).copy()
    variaveis = carregar_variaveis(df)

    if len(variaveis) < 4:
        raise ValueError("Poucas variaveis disponiveis para visualizacao dos clusters.")

    X_proc = matriz_preprocessada(df, variaveis)

    gerar_pca(df, variaveis, X_proc)
    gerar_umap_opcional(df, X_proc)
    gerar_heatmap_centroides(df, variaveis, X_proc)
    gerar_evolucao_clusters(df)
    gerar_evolucao_variavel(
        df,
        "renewables_share_energy",
        "Evolucao da participacao renovavel na matriz energetica",
        "Renovaveis na energia (%)",
        "evolucao_renovaveis_pais.png",
    )
    gerar_evolucao_variavel(
        df,
        "carbon_intensity_elec",
        "Evolucao da intensidade de carbono da eletricidade",
        "Intensidade de carbono da eletricidade",
        "evolucao_intensidade_carbono_pais.png",
    )
    gerar_importancia_permutacao()
    gerar_matriz_confusao()

    print("\nVisualizacoes cientificas concluidas.")


if __name__ == "__main__":
    main()
