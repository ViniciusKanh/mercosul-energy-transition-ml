# -*- coding: utf-8 -*-

"""
Configurações centrais do projeto.

Este arquivo concentra caminhos, países analisados e variáveis candidatas.
A intenção é evitar caminhos absolutos e facilitar a reprodutibilidade.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
REPORTS_DIR = RESULTS_DIR / "reports"

OWID_ENERGY_DATA_URL = "https://owid-public.owid.io/data/energy/owid-energy-data.csv"
OWID_ENERGY_CODEBOOK_URL = "https://owid-public.owid.io/data/energy/owid-energy-codebook.csv"

PAISES_MERCOSUL_ASSOCIADOS = [
    "Brazil",
    "Argentina",
    "Paraguay",
    "Uruguay",
    "Bolivia",
    "Chile",
    "Colombia",
    "Ecuador",
    "Peru",
    "Venezuela",
]

VARIAVEIS_CANDIDATAS = [
    "population",
    "gdp",
    "primary_energy_consumption",
    "energy_per_capita",
    "energy_per_gdp",
    "electricity_generation",
    "electricity_demand",
    "per_capita_electricity",
    "fossil_consumption",
    "fossil_share_energy",
    "fossil_electricity",
    "fossil_share_elec",
    "coal_consumption",
    "coal_share_energy",
    "coal_electricity",
    "coal_share_elec",
    "oil_consumption",
    "oil_share_energy",
    "oil_electricity",
    "oil_share_elec",
    "gas_consumption",
    "gas_share_energy",
    "gas_electricity",
    "gas_share_elec",
    "renewables_consumption",
    "renewables_share_energy",
    "renewables_electricity",
    "renewables_share_elec",
    "low_carbon_consumption",
    "low_carbon_share_energy",
    "low_carbon_electricity",
    "low_carbon_share_elec",
    "hydro_consumption",
    "hydro_share_energy",
    "hydro_electricity",
    "hydro_share_elec",
    "solar_consumption",
    "solar_share_energy",
    "solar_electricity",
    "solar_share_elec",
    "wind_consumption",
    "wind_share_energy",
    "wind_electricity",
    "wind_share_elec",
    "biofuel_consumption",
    "biofuel_share_energy",
    "biofuel_electricity",
    "biofuel_share_elec",
    "nuclear_consumption",
    "nuclear_share_energy",
    "nuclear_electricity",
    "nuclear_share_elec",
    "carbon_intensity_elec",
    "greenhouse_gas_emissions",
    "co2",
    "co2_per_capita",
    "co2_per_unit_energy",
    "coal_co2",
    "gas_co2",
    "oil_co2",
]

VARIAVEIS_CLUSTER_PREFERENCIAIS = [
    "renewables_share_energy",
    "fossil_share_energy",
    "low_carbon_share_energy",
    "renewables_share_elec",
    "fossil_share_elec",
    "low_carbon_share_elec",
    "carbon_intensity_elec",
    "co2_per_capita",
    "co2_per_unit_energy",
    "energy_per_capita",
    "energy_per_gdp",
    "razao_renovavel_fossil_energy",
    "variacao_co2_pct",
    "variacao_renewables_share_energy_pp",
    "media_movel_co2_3anos",
    "tendencia_renovaveis_5anos",
    "indice_transicao_preliminar",
]

RANDOM_STATE = 42
