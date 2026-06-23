# -*- coding: utf-8 -*-

"""
Script 00 - Execução sequencial do pipeline completo.

Uso recomendado apenas depois de testar os scripts individualmente.

Execução:
python scripts/00_executar_pipeline_completo.py
"""

from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = [
    "01_coleta_dados.py",
    "02_auditoria_dados.py",
    "03_engenharia_atributos.py",
    "04_clusterizacao_perfis.py",
    "05_classificacao_ml.py",
    "06_interpretabilidade.py",
    "08_validacao_temporal_sem_vazamento.py",
    "09_robustez_clusterizacao.py",
    "10_interpretacao_perfis_clusters.py",
    "11_visualizacoes_resultados.py",
    "12_estudo_ablacao_variaveis.py",
    "13_exportar_tabelas_latex.py",
    "14_refinamentos_artigo.py",
    "07_relatorio_resultados.py",
]


def main() -> None:
    for script in SCRIPTS:
        caminho = PROJECT_ROOT / "scripts" / script
        print(f"\nExecutando: {script}")
        resultado = subprocess.run([sys.executable, str(caminho)], cwd=PROJECT_ROOT)

        if resultado.returncode != 0:
            raise RuntimeError(f"Falha ao executar {script}. Código: {resultado.returncode}")

    print("\nPipeline completo executado com sucesso.")


if __name__ == "__main__":
    main()
