# -*- coding: utf-8 -*-

"""
Funções utilitárias de entrada, saída e organização de diretórios.
"""

from pathlib import Path
import json
import pandas as pd

from utils.config import RAW_DIR, PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR


def garantir_diretorios() -> None:
    """Cria diretórios principais do projeto, caso ainda não existam."""
    for caminho in [RAW_DIR, PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR]:
        caminho.mkdir(parents=True, exist_ok=True)


def salvar_csv(df: pd.DataFrame, caminho: Path, index: bool = False) -> None:
    """Salva DataFrame em CSV com codificação UTF-8."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, index=index, encoding="utf-8-sig")
    print(f"Arquivo salvo: {caminho}")


def salvar_json(objeto: dict, caminho: Path) -> None:
    """Salva dicionário em JSON legível."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(objeto, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Arquivo salvo: {caminho}")


def verificar_arquivo(caminho: Path) -> None:
    """Interrompe a execução se um arquivo obrigatório não existir."""
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo obrigatório não encontrado: {caminho}\n"
            "Execute os scripts anteriores do pipeline antes de continuar."
        )
