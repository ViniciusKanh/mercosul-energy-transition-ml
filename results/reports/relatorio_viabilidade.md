# Relatorio de Viabilidade Experimental

## Objetivo

Consolidar evidencias iniciais para decidir se o experimento pode avancar para escrita cientifica.

## Resumo dos resultados

- Base: 1099 observacoes, 10 paises, periodo 1900-2025.
- Clusterizacao: Melhor configuracao inicial: `kmeans_k3` com k=3, silhouette=0.367, Davies-Bouldin=1.038 e Calinski-Harabasz=129.932.
- Classificacao original: Melhor modelo: `logistic_regression` com accuracy=1.000, balanced accuracy=1.000 e F1 macro=1.000.
- Validacao sem vazamento: Melhor modelo: `logistic_regression` com accuracy=1.000, balanced accuracy=1.000 e F1 macro=1.000.

## Parecer automatico

- Clusterizacao com separacao inicial aceitavel para exploracao cientifica.
- Classificacao supervisionada reproduz bem os perfis derivados da clusterizacao original.
- Validacao sem vazamento manteve desempenho adequado em teste temporal.

## Observacao metodologica

Os modelos supervisionados reproduzem rotulos derivados de clusterizacao. A narrativa do artigo deve apresentar a classificacao como reproducao de perfis identificados, nao como predicao de uma classe externa real.

## Tabelas principais

### Clusterizacao

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

### Classificacao sem vazamento

| modelo              |   accuracy |   balanced_accuracy |   f1_macro |   f1_weighted |   precision_macro |   recall_macro | estrategia_validacao                          |   qtd_treino |   qtd_teste |   ano_limite_treino |   ano_inicio_teste |   k_clusters_treino |   qtd_variaveis |
|:--------------------|-----------:|--------------------:|-----------:|--------------:|------------------:|---------------:|:----------------------------------------------|-------------:|------------:|--------------------:|-------------------:|--------------------:|----------------:|
| logistic_regression |   1        |            1        |   1        |      1        |          1        |       1        | sem_vazamento_treino_ate_2022_teste_2023_2025 |          230 |          29 |                2022 |               2023 |                   3 |              12 |
| knn                 |   0.965517 |            0.973684 |   0.962677 |      0.965872 |          0.954545 |       0.973684 | sem_vazamento_treino_ate_2022_teste_2023_2025 |          230 |          29 |                2022 |               2023 |                   3 |              12 |
| svm_rbf             |   0.965517 |            0.973684 |   0.962677 |      0.965872 |          0.954545 |       0.973684 | sem_vazamento_treino_ate_2022_teste_2023_2025 |          230 |          29 |                2022 |               2023 |                   3 |              12 |
| random_forest       |   0.965517 |            0.973684 |   0.962677 |      0.965872 |          0.954545 |       0.973684 | sem_vazamento_treino_ate_2022_teste_2023_2025 |          230 |          29 |                2022 |               2023 |                   3 |              12 |
| decision_tree       |   0.724138 |            0.765789 |   0.55     |      0.801724 |          0.607692 |       0.510526 | sem_vazamento_treino_ate_2022_teste_2023_2025 |          230 |          29 |                2022 |               2023 |                   3 |              12 |

### Interpretacao dos clusters

|   cluster_transicao | nome_sugerido                  |   score_baixo_carbono_z |   score_fossil_z |   score_indice_transicao_z | regra_aplicada                                                                                           | variaveis_acima_media                                                                                                                        | variaveis_abaixo_media                                                                                                                         |
|--------------------:|:-------------------------------|------------------------:|-----------------:|---------------------------:|:---------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|
|                   0 | Perfil intermediario           |               -0.146784 |         0.115818 |                  -0.224168 | Indicadores proximos da media geral ou combinacao mista de sinais.                                       | energy_per_gdp (2.50 dp); energy_per_capita (2.03 dp); fossil_share_energy (0.34 dp)                                                         | low_carbon_share_energy (-0.34 dp)                                                                                                             |
|                   1 | Perfil fossil-intensivo        |               -0.880089 |         0.893799 |                  -0.937546 | Indicadores fosseis e/ou intensidade de carbono acima da media, com baixo carbono sem destaque positivo. | fossil_share_elec (0.98 dp); carbon_intensity_elec (0.95 dp); fossil_share_energy (0.74 dp)                                                  | low_carbon_share_elec (-0.98 dp); renewables_share_elec (-0.98 dp); indice_transicao_preliminar (-0.94 dp); renewables_share_energy (-0.76 dp) |
|                   2 | Perfil renovavel/baixo carbono |                0.865768 |        -0.858718 |                   0.845263 | Indicadores renovaveis e de baixo carbono acima da media, sem destaque fossil equivalente.               | low_carbon_share_energy (0.91 dp); renewables_share_energy (0.91 dp); indice_transicao_preliminar (0.85 dp); low_carbon_share_elec (0.84 dp) | fossil_share_energy (-0.91 dp); fossil_share_elec (-0.84 dp); carbon_intensity_elec (-0.83 dp); energy_per_gdp (-0.49 dp)                      |
