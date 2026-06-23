# Relatorio de Resultados - MERCOSUL Descarbonizacao ML

## 1. Objetivo cientifico

Investigar trajetorias de descarbonizacao e transicao energetica em paises associados ao MERCOSUL usando mineracao de dados, clusterizacao, aprendizado de maquina supervisionado e interpretabilidade.

## 2. Fonte e descricao da base

A fonte principal e a Our World in Data Energy Dataset. A base processada possui `1099` observacoes, `10` paises e cobertura temporal de `1900` a `2025`.

Paises analisados:

Argentina, Bolivia, Brazil, Chile, Colombia, Ecuador, Paraguay, Peru, Uruguay, Venezuela

## 3. Cobertura temporal por pais

| country   |   ano_minimo |   ano_maximo |   qtd_anos |   qtd_linhas | pais_previsto_no_projeto   |
|:----------|-------------:|-------------:|-----------:|-------------:|:---------------------------|
| Argentina |         1900 |         2025 |        126 |          126 | True                       |
| Bolivia   |         1900 |         2025 |        126 |          126 | True                       |
| Brazil    |         1900 |         2025 |        126 |          126 | True                       |
| Chile     |         1900 |         2025 |        126 |          126 | True                       |
| Colombia  |         1900 |         2025 |        126 |          126 | True                       |
| Ecuador   |         1900 |         2025 |        126 |          126 | True                       |
| Paraguay  |         1980 |         2025 |         46 |           46 | True                       |
| Peru      |         1900 |         2025 |        126 |          126 | True                       |
| Uruguay   |         1980 |         2025 |         46 |           46 | True                       |
| Venezuela |         1900 |         2024 |        125 |          125 | True                       |

## 4. Variaveis candidatas e variaveis usadas

### Auditoria de variaveis candidatas

| variavel                   | existe_na_base   |   qtd_ausentes |   pct_ausentes |         minimo |          maximo |
|:---------------------------|:-----------------|---------------:|---------------:|---------------:|----------------:|
| population                 | True             |              0 |           0    |    1.45133e+06 |     2.12812e+08 |
| gdp                        | True             |             29 |           2.64 |    1.5372e+09  |     3.18741e+12 |
| primary_energy_consumption | True             |            547 |          49.77 |    6.37992     |  3919.05        |
| energy_per_capita          | True             |            547 |          49.77 | 1632.16        | 35695.2         |
| energy_per_gdp             | True             |            564 |          51.32 |    0.359721    |     4.35672     |
| electricity_generation     | True             |            735 |          66.88 |    3.88        |   750.53        |
| electricity_demand         | True             |            840 |          76.43 |    3.9         |   762.12        |
| per_capita_electricity     | True             |            735 |          66.88 |  450.831       | 10486.4         |
| fossil_consumption         | False            |            nan |         nan    |  nan           |   nan           |
| fossil_share_energy        | True             |            679 |          61.78 |   49.3782      |    98.9988      |
| fossil_electricity         | True             |            810 |          73.7  |    0           |   142.42        |
| fossil_share_elec          | True             |            810 |          73.7  |    0           |    79.2473      |
| coal_consumption           | True             |            679 |          61.78 |    0           |   204.981       |
| coal_share_energy          | True             |            679 |          61.78 |    0           |    24.1377      |
| coal_electricity           | True             |            810 |          73.7  |    0           |    30.26        |
| coal_share_elec            | True             |            810 |          73.7  |    0           |    41.5157      |
| oil_consumption            | True             |            679 |          61.78 |    7.92064     |  1540.81        |
| oil_share_energy           | True             |            679 |          61.78 |   24.4853      |    93.115       |
| oil_electricity            | True             |            810 |          73.7  |    0           |    34.45        |
| oil_share_elec             | True             |            810 |          73.7  |    0           |    44.5932      |
| gas_consumption            | True             |            679 |          61.78 |    0           |   487.021       |
| gas_share_energy           | True             |            679 |          61.78 |    0           |    51.2077      |
| gas_electricity            | True             |            810 |          73.7  |    0           |    95.35        |
| gas_share_elec             | True             |            810 |          73.7  |    0           |    78.172       |
| renewables_consumption     | True             |            679 |          61.78 |    0.691667    |  1898.28        |
| renewables_share_energy    | True             |            679 |          61.78 |    1.00123     |    49.616       |
| renewables_electricity     | True             |            595 |          54.14 |    0.249       |   651.26        |
| renewables_share_elec      | True             |            735 |          66.88 |   20.7527      |   100           |
| low_carbon_consumption     | True             |            679 |          61.78 |    0.691667    |  1936.76        |
| low_carbon_share_energy    | True             |            679 |          61.78 |    1.00123     |    50.6218      |

### Variaveis usadas na clusterizacao

| variavel_usada                      |
|:------------------------------------|
| renewables_share_energy             |
| fossil_share_energy                 |
| low_carbon_share_energy             |
| renewables_share_elec               |
| fossil_share_elec                   |
| low_carbon_share_elec               |
| carbon_intensity_elec               |
| energy_per_capita                   |
| energy_per_gdp                      |
| variacao_renewables_share_energy_pp |
| tendencia_renovaveis_5anos          |
| indice_transicao_preliminar         |

### Variaveis usadas na classificacao

| variavel_usada                      |
|:------------------------------------|
| renewables_share_energy             |
| fossil_share_energy                 |
| low_carbon_share_energy             |
| renewables_share_elec               |
| fossil_share_elec                   |
| low_carbon_share_elec               |
| carbon_intensity_elec               |
| energy_per_capita                   |
| energy_per_gdp                      |
| variacao_renewables_share_energy_pp |
| tendencia_renovaveis_5anos          |
| indice_transicao_preliminar         |

## 5. Resultado da clusterizacao

Melhor configuracao inicial: `kmeans_k3` com k=3, silhouette=0.367, Davies-Bouldin=1.038 e Calinski-Harabasz=129.932.

| modelo_cluster   |   k |   silhouette |   davies_bouldin |   calinski_harabasz |
|:-----------------|----:|-------------:|-----------------:|--------------------:|
| kmeans_k3        |   3 |     0.367382 |          1.0383  |            129.932  |
| agglomerative_k5 |   5 |     0.36453  |          1.08693 |            124.379  |
| kmeans_k5        |   5 |     0.364233 |          1.06897 |            128.93   |
| agglomerative_k3 |   3 |     0.362524 |          1.01904 |            124.882  |
| kmeans_k2        |   2 |     0.360571 |          1.13521 |            175.639  |
| kmeans_k4        |   4 |     0.358713 |          1.09057 |            126.459  |
| agglomerative_k4 |   4 |     0.358167 |          1.0079  |            123.006  |
| agglomerative_k2 |   2 |     0.352565 |          1.16175 |            166.046  |
| gmm_k2           |   2 |     0.2744   |          1.40296 |            105.012  |
| gmm_k5           |   5 |     0.254851 |          1.26247 |             88.0941 |
| gmm_k3           |   3 |     0.232033 |          1.55487 |             96.7726 |
| gmm_k4           |   4 |     0.195958 |          1.42915 |             81.0919 |

## 6. Robustez da clusterizacao

A robustez foi avaliada variando KMeans, AgglomerativeClustering e GaussianMixture, com k entre 2 e 6 e multiplas seeds quando aplicavel.

| modelo_id               | algoritmo     |   k |   seed |   qtd_observacoes |   qtd_variaveis |   silhouette |   davies_bouldin |   calinski_harabasz |
|:------------------------|:--------------|----:|-------:|------------------:|----------------:|-------------:|-----------------:|--------------------:|
| kmeans_k6_seed21        | kmeans        |   6 |     21 |               259 |              12 |     0.380733 |          1.0699  |             127.125 |
| kmeans_k6_seed0         | kmeans        |   6 |      0 |               259 |              12 |     0.380406 |          1.06749 |             127.134 |
| kmeans_k6_seed2024      | kmeans        |   6 |   2024 |               259 |              12 |     0.380406 |          1.06749 |             127.134 |
| kmeans_k6_seed42        | kmeans        |   6 |     42 |               259 |              12 |     0.380169 |          1.07076 |             127.155 |
| kmeans_k6_seed1         | kmeans        |   6 |      1 |               259 |              12 |     0.378914 |          1.06741 |             127.18  |
| kmeans_k6_seed7         | kmeans        |   6 |      7 |               259 |              12 |     0.378914 |          1.06741 |             127.18  |
| kmeans_k6_seed99        | kmeans        |   6 |     99 |               259 |              12 |     0.378914 |          1.06741 |             127.18  |
| kmeans_k6_seed123       | kmeans        |   6 |    123 |               259 |              12 |     0.378914 |          1.06741 |             127.18  |
| agglomerative_k6_seedna | agglomerative |   6 |    nan |               259 |              12 |     0.367442 |          1.06207 |             120.964 |
| kmeans_k3_seed0         | kmeans        |   3 |      0 |               259 |              12 |     0.367382 |          1.0383  |             129.932 |
| kmeans_k3_seed1         | kmeans        |   3 |      1 |               259 |              12 |     0.367382 |          1.0383  |             129.932 |
| kmeans_k3_seed7         | kmeans        |   3 |      7 |               259 |              12 |     0.367382 |          1.0383  |             129.932 |
| kmeans_k3_seed21        | kmeans        |   3 |     21 |               259 |              12 |     0.367382 |          1.0383  |             129.932 |
| kmeans_k3_seed42        | kmeans        |   3 |     42 |               259 |              12 |     0.367382 |          1.0383  |             129.932 |
| kmeans_k3_seed99        | kmeans        |   3 |     99 |               259 |              12 |     0.367382 |          1.0383  |             129.932 |

### Estabilidade entre execucoes

| algoritmo   |   k | modelo_a        | modelo_b           |   seed_a |   seed_b |   adjusted_rand_index |   normalized_mutual_info | observacao                            |
|:------------|----:|:----------------|:-------------------|---------:|---------:|----------------------:|-------------------------:|:--------------------------------------|
| kmeans      |   2 | kmeans_k2_seed0 | kmeans_k2_seed1    |        0 |        1 |                     1 |                        1 | Comparacao par a par entre execucoes. |
| kmeans      |   2 | kmeans_k2_seed0 | kmeans_k2_seed7    |        0 |        7 |                     1 |                        1 | Comparacao par a par entre execucoes. |
| kmeans      |   2 | kmeans_k2_seed0 | kmeans_k2_seed21   |        0 |       21 |                     1 |                        1 | Comparacao par a par entre execucoes. |
| kmeans      |   2 | kmeans_k2_seed0 | kmeans_k2_seed42   |        0 |       42 |                     1 |                        1 | Comparacao par a par entre execucoes. |
| kmeans      |   2 | kmeans_k2_seed0 | kmeans_k2_seed99   |        0 |       99 |                     1 |                        1 | Comparacao par a par entre execucoes. |
| kmeans      |   2 | kmeans_k2_seed0 | kmeans_k2_seed123  |        0 |      123 |                     1 |                        1 | Comparacao par a par entre execucoes. |
| kmeans      |   2 | kmeans_k2_seed0 | kmeans_k2_seed2024 |        0 |     2024 |                     1 |                        1 | Comparacao par a par entre execucoes. |
| kmeans      |   2 | kmeans_k2_seed1 | kmeans_k2_seed7    |        1 |        7 |                     1 |                        1 | Comparacao par a par entre execucoes. |
| kmeans      |   2 | kmeans_k2_seed1 | kmeans_k2_seed21   |        1 |       21 |                     1 |                        1 | Comparacao par a par entre execucoes. |
| kmeans      |   2 | kmeans_k2_seed1 | kmeans_k2_seed42   |        1 |       42 |                     1 |                        1 | Comparacao par a par entre execucoes. |
| kmeans      |   2 | kmeans_k2_seed1 | kmeans_k2_seed99   |        1 |       99 |                     1 |                        1 | Comparacao par a par entre execucoes. |
| kmeans      |   2 | kmeans_k2_seed1 | kmeans_k2_seed123  |        1 |      123 |                     1 |                        1 | Comparacao par a par entre execucoes. |
| kmeans      |   2 | kmeans_k2_seed1 | kmeans_k2_seed2024 |        1 |     2024 |                     1 |                        1 | Comparacao par a par entre execucoes. |
| kmeans      |   2 | kmeans_k2_seed7 | kmeans_k2_seed21   |        7 |       21 |                     1 |                        1 | Comparacao par a par entre execucoes. |
| kmeans      |   2 | kmeans_k2_seed7 | kmeans_k2_seed42   |        7 |       42 |                     1 |                        1 | Comparacao par a par entre execucoes. |

## 7. Interpretacao automatica dos clusters

Os nomes sugeridos abaixo sao calculados a partir de desvios dos indicadores de cada cluster em relacao a media geral. Eles devem ser tratados como rotulos interpretativos, nao como categorias externas observadas.

|   cluster_transicao | nome_sugerido                  |   score_baixo_carbono_z |   score_fossil_z |   score_indice_transicao_z | regra_aplicada                                                                                           | variaveis_acima_media                                                                                                                        | variaveis_abaixo_media                                                                                                                         |
|--------------------:|:-------------------------------|------------------------:|-----------------:|---------------------------:|:---------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|
|                   0 | Perfil intermediario           |               -0.146784 |         0.115818 |                  -0.224168 | Indicadores proximos da media geral ou combinacao mista de sinais.                                       | energy_per_gdp (2.50 dp); energy_per_capita (2.03 dp); fossil_share_energy (0.34 dp)                                                         | low_carbon_share_energy (-0.34 dp)                                                                                                             |
|                   1 | Perfil fossil-intensivo        |               -0.880089 |         0.893799 |                  -0.937546 | Indicadores fosseis e/ou intensidade de carbono acima da media, com baixo carbono sem destaque positivo. | fossil_share_elec (0.98 dp); carbon_intensity_elec (0.95 dp); fossil_share_energy (0.74 dp)                                                  | low_carbon_share_elec (-0.98 dp); renewables_share_elec (-0.98 dp); indice_transicao_preliminar (-0.94 dp); renewables_share_energy (-0.76 dp) |
|                   2 | Perfil renovavel/baixo carbono |                0.865768 |        -0.858718 |                   0.845263 | Indicadores renovaveis e de baixo carbono acima da media, sem destaque fossil equivalente.               | low_carbon_share_energy (0.91 dp); renewables_share_energy (0.91 dp); indice_transicao_preliminar (0.85 dp); low_carbon_share_elec (0.84 dp) | fossil_share_energy (-0.91 dp); fossil_share_elec (-0.84 dp); carbon_intensity_elec (-0.83 dp); energy_per_gdp (-0.49 dp)                      |

### Perfis interpretados

|   cluster_transicao | nome_sugerido                  | caracteristica_principal                                                                                 | variaveis_acima_media                                                                                                                        | variaveis_abaixo_media                                                                                                                         |   media_renewables_share_energy |   desvio_padronizado_renewables_share_energy |   media_fossil_share_energy |   desvio_padronizado_fossil_share_energy |   media_low_carbon_share_energy |   desvio_padronizado_low_carbon_share_energy |   media_renewables_share_elec |   desvio_padronizado_renewables_share_elec |   media_fossil_share_elec |   desvio_padronizado_fossil_share_elec |   media_low_carbon_share_elec |   desvio_padronizado_low_carbon_share_elec |   media_carbon_intensity_elec |   desvio_padronizado_carbon_intensity_elec |   media_energy_per_capita |   desvio_padronizado_energy_per_capita |   media_energy_per_gdp |   desvio_padronizado_energy_per_gdp |   media_variacao_renewables_share_energy_pp |   desvio_padronizado_variacao_renewables_share_energy_pp |   media_tendencia_renovaveis_5anos |   desvio_padronizado_tendencia_renovaveis_5anos |   media_indice_transicao_preliminar |   desvio_padronizado_indice_transicao_preliminar | perfil_transicao        |   qtd_observacoes |   ano_minimo |   ano_maximo |   qtd_paises |   media_renewables_share_energy_original |   media_fossil_share_energy_original |   media_low_carbon_share_energy_original |   media_renewables_share_elec_original |   media_fossil_share_elec_original |   media_low_carbon_share_elec_original |   media_carbon_intensity_elec_original |   media_energy_per_capita_original |   media_energy_per_gdp_original |   media_variacao_renewables_share_energy_pp_original |   media_tendencia_renovaveis_5anos_original |   media_indice_transicao_preliminar_original |
|--------------------:|:-------------------------------|:---------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------:|---------------------------------------------:|----------------------------:|-----------------------------------------:|--------------------------------:|---------------------------------------------:|------------------------------:|-------------------------------------------:|--------------------------:|---------------------------------------:|------------------------------:|-------------------------------------------:|------------------------------:|-------------------------------------------:|--------------------------:|---------------------------------------:|-----------------------:|------------------------------------:|--------------------------------------------:|---------------------------------------------------------:|-----------------------------------:|------------------------------------------------:|------------------------------------:|-------------------------------------------------:|:------------------------|------------------:|-------------:|-------------:|-------------:|-----------------------------------------:|-------------------------------------:|-----------------------------------------:|---------------------------------------:|-----------------------------------:|---------------------------------------:|---------------------------------------:|-----------------------------------:|--------------------------------:|-----------------------------------------------------:|--------------------------------------------:|---------------------------------------------:|
|                   0 | Perfil intermediario           | Indicadores proximos da media geral ou combinacao mista de sinais.                                       | energy_per_gdp (2.50 dp); energy_per_capita (2.03 dp); fossil_share_energy (0.34 dp)                                                         | low_carbon_share_energy (-0.34 dp)                                                                                                             |                         22.7422 |                                    -0.28051  |                     77.2578 |                                 0.337287 |                         22.7422 |                                    -0.337287 |                       67.411  |                                  0.0728626 |                   32.589  |                             -0.0351811 |                       67.411  |                                  0.0351811 |                       256.109 |                                   0.045349 |                   30389.4 |                              2.0257    |               2.52482  |                           2.49943   |                                    0.136387 |                                               -0.0304582 |                          0.272295  |                                        0.077303 |                            0.515593 |                                        -0.224168 | transicao_intermediaria |                23 |         2000 |         2022 |            1 |                                  22.7422 |                              77.2578 |                                  22.7422 |                                67.411  |                            32.589  |                                67.411  |                                256.109 |                            30389.4 |                        2.52482  |                                             0.136387 |                                   0.272295  |                                     0.515593 |
|                   1 | Perfil fossil-intensivo        | Indicadores fosseis e/ou intensidade de carbono acima da media, com baixo carbono sem destaque positivo. | fossil_share_elec (0.98 dp); carbon_intensity_elec (0.95 dp); fossil_share_energy (0.74 dp)                                                  | low_carbon_share_elec (-0.98 dp); renewables_share_elec (-0.98 dp); indice_transicao_preliminar (-0.94 dp); renewables_share_energy (-0.76 dp) |                         18.2855 |                                    -0.75791  |                     81.0086 |                                 0.744392 |                         18.9914 |                                    -0.744392 |                       43.3915 |                                 -0.978518  |                   55.1594 |                              0.982077  |                       44.8406 |                                 -0.982077  |                       389.399 |                                   0.954928 |                   14621.4 |                              0.0239291 |               1.1601   |                          -0.0498878 |                                    0.147826 |                                               -0.0249249 |                         -0.0131613 |                                       -0.210898 |                            0.328646 |                                        -0.937546 | baixa_transicao         |               109 |         2000 |         2025 |            7 |                                  18.2855 |                              81.0086 |                                  18.9914 |                                43.3915 |                            55.1594 |                                44.8406 |                                389.399 |                            14621.4 |                        1.1601   |                                             0.147826 |                                  -0.0131613 |                                     0.328646 |
|                   2 | Perfil renovavel/baixo carbono | Indicadores renovaveis e de baixo carbono acima da media, sem destaque fossil equivalente.               | low_carbon_share_energy (0.91 dp); renewables_share_energy (0.91 dp); indice_transicao_preliminar (0.85 dp); low_carbon_share_elec (0.84 dp) | fossil_share_energy (-0.91 dp); fossil_share_elec (-0.84 dp); carbon_intensity_elec (-0.83 dp); energy_per_gdp (-0.49 dp)                      |                         33.8429 |                                     0.908584 |                     65.749  |                                -0.911844 |                         34.251  |                                     0.911844 |                       84.6314 |                                  0.826635  |                   14.8094 |                             -0.836513  |                       85.1906 |                                  0.836513  |                       128.159 |                                  -0.827796 |                   11182.6 |                             -0.412636  |               0.926345 |                          -0.486555  |                                    0.274952 |                                                0.03657   |                          0.392599  |                                        0.198763 |                            0.795847 |                                         0.845263 | transicao_avancada      |               127 |         2000 |         2025 |            8 |                                  33.8429 |                              65.749  |                                  34.251  |                                84.6314 |                            14.8094 |                                85.1906 |                                128.159 |                            11182.6 |                        0.926345 |                                             0.274952 |                                   0.392599  |                                     0.795847 |

### Paises por cluster no ano mais recente disponivel

| country   |   year |   cluster_transicao | perfil_transicao   | nome_sugerido                  |
|:----------|-------:|--------------------:|:-------------------|:-------------------------------|
| Argentina |   2025 |                   1 | baixa_transicao    | Perfil fossil-intensivo        |
| Bolivia   |   2025 |                   1 | baixa_transicao    | Perfil fossil-intensivo        |
| Peru      |   2025 |                   1 | baixa_transicao    | Perfil fossil-intensivo        |
| Brazil    |   2025 |                   2 | transicao_avancada | Perfil renovavel/baixo carbono |
| Chile     |   2025 |                   2 | transicao_avancada | Perfil renovavel/baixo carbono |
| Colombia  |   2025 |                   2 | transicao_avancada | Perfil renovavel/baixo carbono |
| Ecuador   |   2025 |                   2 | transicao_avancada | Perfil renovavel/baixo carbono |
| Paraguay  |   2025 |                   2 | transicao_avancada | Perfil renovavel/baixo carbono |
| Uruguay   |   2025 |                   2 | transicao_avancada | Perfil renovavel/baixo carbono |
| Venezuela |   2024 |                   2 | transicao_avancada | Perfil renovavel/baixo carbono |

### Cautela sobre o cluster intermediario

O cluster intermediario contem 23 registros, cobre 1 pais(es) e aparece entre 2000 e 2022. Como esta associado apenas a Venezuela, deve ser tratado com cautela: ele pode representar uma trajetoria nacional especifica, e nao um padrao regional amplo.

Tabela detalhada salva em `results/tables/cluster_intermediario_detalhado.csv`.

| country   |   year | perfil_transicao        |   renewables_share_energy |   fossil_share_energy |   renewables_share_elec |   fossil_share_elec |   carbon_intensity_elec |   energy_per_capita |   energy_per_gdp |   indice_transicao_preliminar |
|:----------|-------:|:------------------------|--------------------------:|----------------------:|------------------------:|--------------------:|------------------------:|--------------------:|-----------------:|------------------------------:|
| Venezuela |   2000 | transicao_intermediaria |                   22.9622 |               77.0378 |                 73.754  |             26.246  |                  212.74 |             31476.4 |          2.33776 |                      0.55705  |
| Venezuela |   2001 | transicao_intermediaria |                   20.7048 |               79.2952 |                 67.0736 |             32.9264 |                  261.57 |             32706.3 |          2.37976 |                      0.491368 |
| Venezuela |   2002 | transicao_intermediaria |                   20.2572 |               79.7428 |                 64.8122 |             35.1878 |                  274.14 |             32074.7 |          2.59819 |                      0.47411  |
| Venezuela |   2003 | transicao_intermediaria |                   22.8002 |               77.1998 |                 65.8293 |             34.1707 |                  263.73 |             28348.3 |          2.5195  |                      0.507819 |
| Venezuela |   2004 | transicao_intermediaria |                   23.5925 |               76.4075 |                 71.101  |             28.899  |                  228.62 |             31015.1 |          2.359   |                      0.548065 |
| Venezuela |   2005 | transicao_intermediaria |                   24.7975 |               75.2025 |                 73.2802 |             26.7198 |                  211.22 |             31702.9 |          2.2139  |                      0.574913 |
| Venezuela |   2006 | transicao_intermediaria |                   23.6613 |               76.3387 |                 73.8997 |             26.1003 |                  207.21 |             34305.9 |          2.20338 |                      0.567076 |
| Venezuela |   2007 | transicao_intermediaria |                   22.9629 |               77.0371 |                 72.7257 |             27.2743 |                  217.67 |             35287   |          2.11056 |                      0.55163  |
| Venezuela |   2008 | transicao_intermediaria |                   23.2173 |               76.7827 |                 72.7913 |             27.2087 |                  215.68 |             35657.4 |          2.04863 |                      0.555308 |
| Venezuela |   2009 | transicao_intermediaria |                   22.6124 |               77.3876 |                 71.8909 |             28.1091 |                  222.13 |             35543   |          2.13084 |                      0.543452 |
| Venezuela |   2010 | transicao_intermediaria |                   21.1514 |               78.8486 |                 67.493  |             32.507  |                  252.02 |             33214.4 |          2.00382 |                      0.501467 |
| Venezuela |   2011 | transicao_intermediaria |                   21.6311 |               78.3689 |                 70.9308 |             29.0692 |                  227.11 |             34603.8 |          2.00903 |                      0.528228 |

### Interpretacao dos paises no ano mais recente com indicadores completos

A tabela abaixo organiza os indicadores centrais de cada pais no ano mais recente em que todos os indicadores exigidos estao disponiveis, junto ao perfil de transicao energetica atribuido pela clusterizacao. Quando nenhum ano possui todos os indicadores completos, a tabela registra essa limitacao explicitamente.

| country   |   year | perfil_transicao   | nome_sugerido                  |   renewables_share_energy |   fossil_share_energy |   renewables_share_elec |   fossil_share_elec |   carbon_intensity_elec |   indice_transicao_preliminar | observacao_completude                      |
|:----------|-------:|:-------------------|:-------------------------------|--------------------------:|----------------------:|------------------------:|--------------------:|------------------------:|------------------------------:|:-------------------------------------------|
| Argentina |   2024 | baixa_transicao    | Perfil fossil-intensivo        |                   14.803  |               82.5832 |                 34.3846 |             58.7634 |                  344.83 |                      0.326071 | Completo                                   |
| Bolivia   |    nan | nan                | nan                            |                  nan      |              nan      |                nan      |            nan      |                  nan    |                    nan        | Sem ano com todos os indicadores completos |
| Brazil    |   2024 | transicao_avancada | Perfil renovavel/baixo carbono |                   49.616  |               49.3782 |                 87.3331 |             10.5509 |                  106.06 |                      0.930231 | Completo                                   |
| Chile     |   2024 | transicao_avancada | Perfil renovavel/baixo carbono |                   32.2544 |               67.7456 |                 70.0406 |             29.9594 |                  259.87 |                      0.619064 | Completo                                   |
| Colombia  |   2024 | baixa_transicao    | Perfil fossil-intensivo        |                   26.0929 |               73.9071 |                 61.3417 |             38.6583 |                  298.3  |                      0.511873 | Completo                                   |
| Ecuador   |   2024 | transicao_avancada | Perfil renovavel/baixo carbono |                   25.3576 |               74.6424 |                 72.4434 |             27.5566 |                  203.61 |                      0.581345 | Completo                                   |
| Paraguay  |    nan | nan                | nan                            |                  nan      |              nan      |                nan      |            nan      |                  nan    |                    nan        | Sem ano com todos os indicadores completos |
| Peru      |   2024 | baixa_transicao    | Perfil fossil-intensivo        |                   26.6562 |               73.3438 |                 60.1113 |             39.8887 |                  258.73 |                      0.531251 | Completo                                   |
| Uruguay   |    nan | nan                | nan                            |                  nan      |              nan      |                nan      |            nan      |                  nan    |                    nan        | Sem ano com todos os indicadores completos |
| Venezuela |   2024 | transicao_avancada | Perfil renovavel/baixo carbono |                   25.8403 |               74.1597 |                 91.1394 |              8.8606 |                   85.86 |                      0.697389 | Completo                                   |

## 8. Resultado da classificacao supervisionada

Melhor modelo: `logistic_regression` com accuracy=1.000, balanced accuracy=1.000 e F1 macro=1.000.

| modelo              |   accuracy |   balanced_accuracy |   f1_macro |   f1_weighted |   precision_macro |   recall_macro | estrategia_validacao                     |   qtd_treino |   qtd_teste |
|:--------------------|-----------:|--------------------:|-----------:|--------------:|------------------:|---------------:|:-----------------------------------------|-------------:|------------:|
| logistic_regression |   1        |            1        |   1        |      1        |          1        |       1        | temporal_treino_ate_2022_teste_2023_2025 |          230 |          29 |
| knn                 |   0.965517 |            0.973684 |   0.962677 |      0.965872 |          0.954545 |       0.973684 | temporal_treino_ate_2022_teste_2023_2025 |          230 |          29 |
| svm_rbf             |   0.965517 |            0.973684 |   0.962677 |      0.965872 |          0.954545 |       0.973684 | temporal_treino_ate_2022_teste_2023_2025 |          230 |          29 |
| random_forest       |   0.965517 |            0.95     |   0.960864 |      0.965052 |          0.975    |       0.95     | temporal_treino_ate_2022_teste_2023_2025 |          230 |          29 |
| decision_tree       |   0.724138 |            0.765789 |   0.55     |      0.801724 |          0.607692 |       0.510526 | temporal_treino_ate_2022_teste_2023_2025 |          230 |          29 |

Importante: o modelo supervisionado aprende a reproduzir perfis de transicao energetica identificados por clusterizacao nao supervisionada. Portanto, o alvo nao e uma classe externa real.

### Diagnostico da matriz de confusao

As seguintes classes possuem linha ou coluna zerada na matriz de teste temporal: Intermediário. Em especial, se o perfil intermediario estiver zerado, isso indica que ele nao apareceu no conjunto de teste e deve ser interpretado como limitacao da amostra, nao como ausencia substantiva do perfil.

| modelo              | classe                  | presente_no_teste   |   qtd_real_teste |   qtd_predito | linha_zerada   | coluna_zerada   | observacao                                                                                                 |
|:--------------------|:------------------------|:--------------------|-----------------:|--------------:|:---------------|:----------------|:-----------------------------------------------------------------------------------------------------------|
| logistic_regression | Intermediário           | False               |                0 |             0 | True           | True            | Perfil ausente no conjunto de teste temporal; interpretar a linha/coluna zerada como limitação da amostra. |
| logistic_regression | Fóssil-intensivo        | True                |               10 |            10 | False          | False           | Perfil presente no conjunto de teste temporal.                                                             |
| logistic_regression | Renovável/baixo carbono | True                |               19 |            19 | False          | False           | Perfil presente no conjunto de teste temporal.                                                             |

## 9. Validacao temporal sem vazamento

Na validacao sem vazamento, o imputador, o scaler e o KMeans sao ajustados apenas no treino ate 2022. Os anos de 2023 em diante sao atribuidos ao centroide treinado mais proximo, e os classificadores sao treinados apenas com o periodo de treino.

Melhor modelo: `logistic_regression` com accuracy=1.000, balanced accuracy=1.000 e F1 macro=1.000.

| modelo              |   accuracy |   balanced_accuracy |   f1_macro |   f1_weighted |   precision_macro |   recall_macro | estrategia_validacao                          |   qtd_treino |   qtd_teste |   ano_limite_treino |   ano_inicio_teste |   k_clusters_treino |   qtd_variaveis |
|:--------------------|-----------:|--------------------:|-----------:|--------------:|------------------:|---------------:|:----------------------------------------------|-------------:|------------:|--------------------:|-------------------:|--------------------:|----------------:|
| logistic_regression |   1        |            1        |   1        |      1        |          1        |       1        | sem_vazamento_treino_ate_2022_teste_2023_2025 |          230 |          29 |                2022 |               2023 |                   3 |              12 |
| knn                 |   0.965517 |            0.973684 |   0.962677 |      0.965872 |          0.954545 |       0.973684 | sem_vazamento_treino_ate_2022_teste_2023_2025 |          230 |          29 |                2022 |               2023 |                   3 |              12 |
| svm_rbf             |   0.965517 |            0.973684 |   0.962677 |      0.965872 |          0.954545 |       0.973684 | sem_vazamento_treino_ate_2022_teste_2023_2025 |          230 |          29 |                2022 |               2023 |                   3 |              12 |
| random_forest       |   0.965517 |            0.973684 |   0.962677 |      0.965872 |          0.954545 |       0.973684 | sem_vazamento_treino_ate_2022_teste_2023_2025 |          230 |          29 |                2022 |               2023 |                   3 |              12 |
| decision_tree       |   0.724138 |            0.765789 |   0.55     |      0.801724 |          0.607692 |       0.510526 | sem_vazamento_treino_ate_2022_teste_2023_2025 |          230 |          29 |                2022 |               2023 |                   3 |              12 |

### Comparacao com a validacao original

| cenario                                                      | modelo              |   accuracy |   balanced_accuracy |   f1_macro |   f1_weighted | observacao                                                             |
|:-------------------------------------------------------------|:--------------------|-----------:|--------------------:|-----------:|--------------:|:-----------------------------------------------------------------------|
| validacao_original_com_rotulos_clusterizados_em_toda_amostra | logistic_regression |          1 |                   1 |          1 |             1 | Referencia historica; pode conter vazamento na construcao dos rotulos. |
| validacao_temporal_sem_vazamento                             | logistic_regression |          1 |                   1 |          1 |             1 | Scaler, imputador e KMeans ajustados apenas no treino ate 2022.        |

## 10. Estudo de ablacao

O estudo de ablacao compara cenarios com todas as variaveis, sem o indice preliminar, apenas energia, apenas eletricidade e sem variaveis derivadas do projeto.

### Ablacao na classificacao

| modelo              |   accuracy |   balanced_accuracy |   f1_macro |   f1_weighted |   precision_macro |   recall_macro | cenario                           |   qtd_variaveis | variaveis                                                                                                                                                                                                                                                                                      | estrategia_validacao                          |   qtd_treino |   qtd_teste | observacao                                                     |
|:--------------------|-----------:|--------------------:|-----------:|--------------:|------------------:|---------------:|:----------------------------------|----------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------|-------------:|------------:|:---------------------------------------------------------------|
| logistic_regression |   1        |            1        |   1        |      1        |          1        |       1        | A_todas_variaveis                 |              12 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; renewables_share_elec; fossil_share_elec; low_carbon_share_elec; carbon_intensity_elec; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp; tendencia_renovaveis_5anos; indice_transicao_preliminar | temporal_treino_ate_2022_teste_2023_em_diante |          230 |          29 | Classificacao dos rotulos derivados da clusterizacao original. |
| knn                 |   0.965517 |            0.973684 |   0.962677 |      0.965872 |          0.954545 |       0.973684 | A_todas_variaveis                 |              12 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; renewables_share_elec; fossil_share_elec; low_carbon_share_elec; carbon_intensity_elec; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp; tendencia_renovaveis_5anos; indice_transicao_preliminar | temporal_treino_ate_2022_teste_2023_em_diante |          230 |          29 | Classificacao dos rotulos derivados da clusterizacao original. |
| random_forest       |   0.965517 |            0.95     |   0.960864 |      0.965052 |          0.975    |       0.95     | A_todas_variaveis                 |              12 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; renewables_share_elec; fossil_share_elec; low_carbon_share_elec; carbon_intensity_elec; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp; tendencia_renovaveis_5anos; indice_transicao_preliminar | temporal_treino_ate_2022_teste_2023_em_diante |          230 |          29 | Classificacao dos rotulos derivados da clusterizacao original. |
| logistic_regression |   1        |            1        |   1        |      1        |          1        |       1        | B_sem_indice_transicao_preliminar |              11 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; renewables_share_elec; fossil_share_elec; low_carbon_share_elec; carbon_intensity_elec; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp; tendencia_renovaveis_5anos                              | temporal_treino_ate_2022_teste_2023_em_diante |          230 |          29 | Classificacao dos rotulos derivados da clusterizacao original. |
| knn                 |   0.965517 |            0.973684 |   0.962677 |      0.965872 |          0.954545 |       0.973684 | B_sem_indice_transicao_preliminar |              11 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; renewables_share_elec; fossil_share_elec; low_carbon_share_elec; carbon_intensity_elec; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp; tendencia_renovaveis_5anos                              | temporal_treino_ate_2022_teste_2023_em_diante |          230 |          29 | Classificacao dos rotulos derivados da clusterizacao original. |
| random_forest       |   0.931034 |            0.923684 |   0.923684 |      0.931034 |          0.923684 |       0.923684 | B_sem_indice_transicao_preliminar |              11 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; renewables_share_elec; fossil_share_elec; low_carbon_share_elec; carbon_intensity_elec; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp; tendencia_renovaveis_5anos                              | temporal_treino_ate_2022_teste_2023_em_diante |          230 |          29 | Classificacao dos rotulos derivados da clusterizacao original. |
| knn                 |   0.517241 |            0.584211 |   0.516667 |      0.511494 |          0.588889 |       0.584211 | C_apenas_variaveis_energia        |               6 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp                                                                                                                                                  | temporal_treino_ate_2022_teste_2023_em_diante |          230 |          29 | Classificacao dos rotulos derivados da clusterizacao original. |
| logistic_regression |   0.517241 |            0.607895 |   0.512019 |      0.496353 |          0.633117 |       0.607895 | C_apenas_variaveis_energia        |               6 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp                                                                                                                                                  | temporal_treino_ate_2022_teste_2023_em_diante |          230 |          29 | Classificacao dos rotulos derivados da clusterizacao original. |
| random_forest       |   0.517241 |            0.607895 |   0.512019 |      0.496353 |          0.633117 |       0.607895 | C_apenas_variaveis_energia        |               6 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp                                                                                                                                                  | temporal_treino_ate_2022_teste_2023_em_diante |          230 |          29 | Classificacao dos rotulos derivados da clusterizacao original. |
| random_forest       |   0.862069 |            0.871053 |   0.612086 |      0.909054 |          0.647059 |       0.580702 | D_apenas_variaveis_eletricidade   |               4 | renewables_share_elec; fossil_share_elec; low_carbon_share_elec; carbon_intensity_elec                                                                                                                                                                                                         | temporal_treino_ate_2022_teste_2023_em_diante |          230 |          29 | Classificacao dos rotulos derivados da clusterizacao original. |
| knn                 |   0.827586 |            0.821053 |   0.56899  |      0.857017 |          0.592593 |       0.547368 | D_apenas_variaveis_eletricidade   |               4 | renewables_share_elec; fossil_share_elec; low_carbon_share_elec; carbon_intensity_elec                                                                                                                                                                                                         | temporal_treino_ate_2022_teste_2023_em_diante |          230 |          29 | Classificacao dos rotulos derivados da clusterizacao original. |
| logistic_regression |   0.724138 |            0.718421 |   0.557338 |      0.83988  |          0.666667 |       0.478947 | D_apenas_variaveis_eletricidade   |               4 | renewables_share_elec; fossil_share_elec; low_carbon_share_elec; carbon_intensity_elec                                                                                                                                                                                                         | temporal_treino_ate_2022_teste_2023_em_diante |          230 |          29 | Classificacao dos rotulos derivados da clusterizacao original. |
| logistic_regression |   0.965517 |            0.973684 |   0.962677 |      0.965872 |          0.954545 |       0.973684 | E_sem_variaveis_derivadas         |               9 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; renewables_share_elec; fossil_share_elec; low_carbon_share_elec; carbon_intensity_elec; energy_per_capita; energy_per_gdp                                                                                               | temporal_treino_ate_2022_teste_2023_em_diante |          230 |          29 | Classificacao dos rotulos derivados da clusterizacao original. |
| random_forest       |   0.931034 |            0.923684 |   0.923684 |      0.931034 |          0.923684 |       0.923684 | E_sem_variaveis_derivadas         |               9 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; renewables_share_elec; fossil_share_elec; low_carbon_share_elec; carbon_intensity_elec; energy_per_capita; energy_per_gdp                                                                                               | temporal_treino_ate_2022_teste_2023_em_diante |          230 |          29 | Classificacao dos rotulos derivados da clusterizacao original. |
| knn                 |   0.896552 |            0.897368 |   0.888031 |      0.897617 |          0.881313 |       0.897368 | E_sem_variaveis_derivadas         |               9 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; renewables_share_elec; fossil_share_elec; low_carbon_share_elec; carbon_intensity_elec; energy_per_capita; energy_per_gdp                                                                                               | temporal_treino_ate_2022_teste_2023_em_diante |          230 |          29 | Classificacao dos rotulos derivados da clusterizacao original. |

### Ablacao na clusterizacao

| cenario                           |   k |   qtd_variaveis | variaveis                                                                                                                                                                                                                                                                                      |   silhouette |   davies_bouldin |   calinski_harabasz | observacao                                               |
|:----------------------------------|----:|----------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------:|-----------------:|--------------------:|:---------------------------------------------------------|
| A_todas_variaveis                 |   6 |              12 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; renewables_share_elec; fossil_share_elec; low_carbon_share_elec; carbon_intensity_elec; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp; tendencia_renovaveis_5anos; indice_transicao_preliminar |     0.378914 |         1.06741  |             127.18  | KMeans ajustado para o conjunto de variaveis do cenario. |
| A_todas_variaveis                 |   3 |              12 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; renewables_share_elec; fossil_share_elec; low_carbon_share_elec; carbon_intensity_elec; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp; tendencia_renovaveis_5anos; indice_transicao_preliminar |     0.367382 |         1.0383   |             129.932 | KMeans ajustado para o conjunto de variaveis do cenario. |
| A_todas_variaveis                 |   5 |              12 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; renewables_share_elec; fossil_share_elec; low_carbon_share_elec; carbon_intensity_elec; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp; tendencia_renovaveis_5anos; indice_transicao_preliminar |     0.364233 |         1.06897  |             128.93  | KMeans ajustado para o conjunto de variaveis do cenario. |
| A_todas_variaveis                 |   2 |              12 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; renewables_share_elec; fossil_share_elec; low_carbon_share_elec; carbon_intensity_elec; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp; tendencia_renovaveis_5anos; indice_transicao_preliminar |     0.360571 |         1.13521  |             175.639 | KMeans ajustado para o conjunto de variaveis do cenario. |
| A_todas_variaveis                 |   4 |              12 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; renewables_share_elec; fossil_share_elec; low_carbon_share_elec; carbon_intensity_elec; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp; tendencia_renovaveis_5anos; indice_transicao_preliminar |     0.359111 |         1.07459  |             126.526 | KMeans ajustado para o conjunto de variaveis do cenario. |
| B_sem_indice_transicao_preliminar |   6 |              11 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; renewables_share_elec; fossil_share_elec; low_carbon_share_elec; carbon_intensity_elec; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp; tendencia_renovaveis_5anos                              |     0.36328  |         1.07709  |             117.618 | KMeans ajustado para o conjunto de variaveis do cenario. |
| B_sem_indice_transicao_preliminar |   4 |              11 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; renewables_share_elec; fossil_share_elec; low_carbon_share_elec; carbon_intensity_elec; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp; tendencia_renovaveis_5anos                              |     0.360104 |         1.05717  |             120.776 | KMeans ajustado para o conjunto de variaveis do cenario. |
| B_sem_indice_transicao_preliminar |   5 |              11 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; renewables_share_elec; fossil_share_elec; low_carbon_share_elec; carbon_intensity_elec; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp; tendencia_renovaveis_5anos                              |     0.345764 |         1.11604  |             117.566 | KMeans ajustado para o conjunto de variaveis do cenario. |
| B_sem_indice_transicao_preliminar |   2 |              11 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; renewables_share_elec; fossil_share_elec; low_carbon_share_elec; carbon_intensity_elec; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp; tendencia_renovaveis_5anos                              |     0.344704 |         1.19388  |             157.337 | KMeans ajustado para o conjunto de variaveis do cenario. |
| B_sem_indice_transicao_preliminar |   3 |              11 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; renewables_share_elec; fossil_share_elec; low_carbon_share_elec; carbon_intensity_elec; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp; tendencia_renovaveis_5anos                              |     0.330275 |         1.23859  |             120.965 | KMeans ajustado para o conjunto de variaveis do cenario. |
| C_apenas_variaveis_energia        |   5 |               6 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp                                                                                                                                                  |     0.449666 |         0.992052 |             154.414 | KMeans ajustado para o conjunto de variaveis do cenario. |
| C_apenas_variaveis_energia        |   4 |               6 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp                                                                                                                                                  |     0.439662 |         0.893814 |             160.831 | KMeans ajustado para o conjunto de variaveis do cenario. |
| C_apenas_variaveis_energia        |   6 |               6 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp                                                                                                                                                  |     0.4369   |         0.941411 |             149.673 | KMeans ajustado para o conjunto de variaveis do cenario. |
| C_apenas_variaveis_energia        |   3 |               6 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp                                                                                                                                                  |     0.428972 |         0.968883 |             151.477 | KMeans ajustado para o conjunto de variaveis do cenario. |
| C_apenas_variaveis_energia        |   2 |               6 | renewables_share_energy; fossil_share_energy; low_carbon_share_energy; energy_per_capita; energy_per_gdp; variacao_renewables_share_energy_pp                                                                                                                                                  |     0.377815 |         1.23348  |             122.506 | KMeans ajustado para o conjunto de variaveis do cenario. |

## 11. Importancia das variaveis

| variavel                            |   importancia_media |   importancia_desvio | modelo              |
|:------------------------------------|--------------------:|---------------------:|:--------------------|
| tendencia_renovaveis_5anos          |           0.0755447 |            0.027786  | logistic_regression |
| carbon_intensity_elec               |           0.0723268 |            0.0423381 | logistic_regression |
| fossil_share_elec                   |           0.0719582 |            0.0383833 | logistic_regression |
| low_carbon_share_elec               |           0.0719582 |            0.0383833 | logistic_regression |
| renewables_share_elec               |           0.070345  |            0.0411297 | logistic_regression |
| energy_per_capita                   |           0.0579105 |            0.129503  | logistic_regression |
| indice_transicao_preliminar         |           0.0548004 |            0.0345874 | logistic_regression |
| renewables_share_energy             |           0.0206214 |            0.0238002 | logistic_regression |
| low_carbon_share_energy             |           0.0156545 |            0.0191728 | logistic_regression |
| fossil_share_energy                 |           0.0156545 |            0.0191728 | logistic_regression |
| variacao_renewables_share_energy_pp |           0         |            0         | logistic_regression |
| energy_per_gdp                      |           0         |            0         | logistic_regression |

## 12. Figuras geradas

As figuras cientificas foram salvas em `results/figures/`, incluindo PCA dos clusters, heatmap dos centroides, evolucao temporal dos clusters, evolucao de renovaveis, evolucao da intensidade de carbono, importancia por permutacao e matriz de confusao.

### Figuras recomendadas para o artigo

- `results/figures/pca_clusters.png`
- `results/figures/heatmap_clusters_pais_ano.png`
- `results/figures/heatmap_centroides_artigo.png`
- `results/figures/importancia_permutacao_top15.png`
- `results/figures/matriz_confusao_artigo.png`
- `results/figures/evolucao_intensidade_carbono_paises_chave.png`
- `results/figures/evolucao_renovaveis_paises_chave.png`

## 13. Limitacoes metodologicas

- Os rotulos supervisionados sao derivados da clusterizacao e nao representam classes externas observadas.
- A qualidade dos clusters depende das variaveis selecionadas, do tratamento de ausencias e do recorte temporal.
- O desempenho perfeito ou muito alto na classificacao deve ser interpretado como capacidade de reproduzir rotulos derivados, nao como prova causal de descarbonizacao.
- Algumas variaveis possuem lacunas historicas, especialmente nos anos mais antigos.
- O indice de transicao preliminar e uma variavel composta; por isso, seus efeitos devem ser discutidos com apoio do estudo de ablacao.
- O cluster intermediario e restrito a uma trajetoria nacional especifica quando associado a apenas um pais, devendo ser interpretado com cautela.
- A matriz de confusao da validacao temporal pode nao conter o perfil intermediario no conjunto de teste, produzindo linha ou coluna zerada para essa classe.
- A base OWID pode sofrer revisoes, entao execucoes futuras podem alterar pequenos valores e metricas.

## 14. Proximos passos

- Revisar as figuras para selecionar as que entram no artigo.
- Priorizar resultados sem vazamento na narrativa metodologica.
- Usar os nomes de clusters como hipoteses interpretativas e validar com literatura energetica regional.
- Comparar trajetorias nacionais recentes com eventos de politica energetica e contexto economico.
- Preparar tabelas finais em LaTeX a partir de `results/latex/`.

## 15. Parecer automatico

- Clusterizacao com separacao inicial aceitavel para exploracao cientifica.
- Classificacao supervisionada reproduz bem os perfis derivados da clusterizacao original.
- Validacao sem vazamento manteve desempenho adequado em teste temporal.
