# -*- coding: utf-8 -*-

"""
Script 01 - Coleta dos dados brutos.

Objetivo:
Baixar a base OWID Energy Dataset e seu dicionário de variáveis.

Execução:
python scripts/01_coleta_dados.py
python scripts/01_coleta_dados.py --force
"""

from pathlib import Path
import argparse
import datetime as dt
import sys
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from utils.config import RAW_DIR, OWID_ENERGY_DATA_URL, OWID_ENERGY_CODEBOOK_URL  # noqa: E402
from utils.io_utils import garantir_diretorios, salvar_json  # noqa: E402


def baixar_arquivo(url: str, destino: Path, force: bool = False) -> None:
    """Baixa um arquivo se ele ainda não existir ou se force=True."""
    if destino.exists() and not force:
        print(f"Arquivo já existe. Download ignorado: {destino}")
        return

    print(f"Baixando: {url}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destino)
    print(f"Arquivo salvo: {destino}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Coleta da base OWID Energy Dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Força novo download mesmo se os arquivos já existirem.",
    )
    args = parser.parse_args()

    garantir_diretorios()

    caminho_base = RAW_DIR / "owid-energy-data.csv"
    caminho_codebook = RAW_DIR / "owid-energy-codebook.csv"

    baixar_arquivo(OWID_ENERGY_DATA_URL, caminho_base, force=args.force)
    baixar_arquivo(OWID_ENERGY_CODEBOOK_URL, caminho_codebook, force=args.force)

    metadados = {
        "data_execucao": dt.datetime.now().isoformat(timespec="seconds"),
        "fonte_principal": "Our World in Data Energy Dataset",
        "url_base": OWID_ENERGY_DATA_URL,
        "url_codebook": OWID_ENERGY_CODEBOOK_URL,
        "arquivo_base": str(caminho_base.relative_to(PROJECT_ROOT)),
        "arquivo_codebook": str(caminho_codebook.relative_to(PROJECT_ROOT)),
    }

    salvar_json(metadados, RAW_DIR / "metadados_coleta_owid.json")
    print("\nColeta concluída com sucesso.")


if __name__ == "__main__":
    main()
