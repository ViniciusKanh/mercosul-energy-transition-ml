---
name: mercosul-descarbonizacao-ml
description: Atuar como especialista técnico-científico no projeto mercosul-descarbonizacao-ml. Use quando Codex precisar desenvolver, refatorar, validar metodologicamente, gerar resultados, criar visualizações científicas, exportar tabelas LaTeX ou preparar escrita acadêmica sobre trajetórias de descarbonização e transição energética em países associados ao MERCOSUL usando mineração de dados, clusterização, aprendizado de máquina supervisionado e interpretabilidade.
---

# mercosul-descarbonizacao-ml

## Objetivo da skill

Atuar como especialista técnico-científico no projeto `mercosul-descarbonizacao-ml`, cujo objetivo é investigar trajetórias de descarbonização e transição energética em países associados ao MERCOSUL usando mineração de dados, clusterização, aprendizado de máquina supervisionado e interpretabilidade.

Apoiar desenvolvimento, refatoração, validação metodológica, geração de resultados, criação de visualizações científicas, exportação de tabelas LaTeX e preparação do projeto para escrita acadêmica.

## Contexto científico

Usar dados energéticos da base Our World in Data Energy Dataset para analisar países associados ao MERCOSUL:

- Argentina
- Bolivia
- Brazil
- Chile
- Colombia
- Ecuador
- Paraguay
- Peru
- Uruguay
- Venezuela

Identificar perfis de transição energética e descarbonização a partir de variáveis como:

- participação de renováveis na matriz energética;
- participação fóssil na matriz energética;
- participação de baixo carbono;
- participação de renováveis na eletricidade;
- intensidade de carbono da eletricidade;
- energia per capita;
- energia por PIB;
- tendência de renováveis;
- variações temporais de emissões e matriz energética.

## Estrutura atual esperada

Considerar a seguinte estrutura-base do projeto:

```text
scripts/
├── 00_executar_pipeline_completo.py
├── 01_coleta_dados.py
├── 02_auditoria_dados.py
├── 03_engenharia_atributos.py
├── 04_clusterizacao_perfis.py
├── 05_classificacao_ml.py
├── 06_interpretabilidade.py
└── 07_relatorio_resultados.py
```

Considerar estes diretórios principais:

```text
data/raw/
data/processed/
results/tables/
results/figures/
results/reports/
results/latex/
_backups/
```

## Pipeline metodológico

Seguir esta lógica de pipeline:

1. Coletar os dados.
2. Auditar a base.
3. Filtrar os países.
4. Fazer engenharia de atributos.
5. Clusterizar os perfis de transição energética.
6. Classificar supervisionadamente os perfis.
7. Interpretar os modelos.
8. Validar temporalmente sem vazamento.
9. Avaliar robustez da clusterização.
10. Interpretar os clusters.
11. Criar visualizações científicas.
12. Executar estudo de ablação.
13. Exportar tabelas LaTeX.
14. Gerar relatório final.

## Regras metodológicas obrigatórias

### 1. Evitar vazamento de dados

Nunca ajustar scaler, imputador, PCA, clusterizador ou qualquer transformador usando dados de teste.

Para validação temporal:

- usar treino com anos até 2022;
- usar teste com anos de 2023 em diante;
- ajustar scaler apenas no treino;
- ajustar clusterização apenas no treino;
- atribuir dados de teste ao cluster mais próximo;
- treinar classificadores somente no treino.

### 2. Tratar rótulos com cautela

Tratar rótulos da classificação como derivados da clusterização. Não descrever a classificação como predição de uma classe real externa.

Usar formulação semelhante a:

> O modelo supervisionado aprende a reproduzir perfis de transição energética identificados por clusterização não supervisionada.

### 3. Exigir robustez

Sempre que possível, avaliar:

- diferentes valores de `k`;
- diferentes algoritmos;
- diferentes seeds;
- silhouette;
- Davies-Bouldin;
- Calinski-Harabasz;
- Adjusted Rand Index;
- Normalized Mutual Information.

### 4. Justificar variáveis derivadas

Testar variáveis compostas, como `indice_transicao_preliminar`, em estudos de ablação.

Gerar pelo menos dois cenários:

- com índice;
- sem índice.

## Scripts recomendados adicionais

Considerar desejável a criação destes scripts:

```text
scripts/08_validacao_temporal_sem_vazamento.py
scripts/09_robustez_clusterizacao.py
scripts/10_interpretacao_perfis_clusters.py
scripts/11_visualizacoes_resultados.py
scripts/12_estudo_ablacao_variaveis.py
scripts/13_exportar_tabelas_latex.py
```

## Saídas esperadas

### Tabelas

```text
results/tables/metricas_clusterizacao.csv
results/tables/perfis_clusters.csv
results/tables/variaveis_usadas_clusterizacao.csv
results/tables/metricas_classificacao.csv
results/tables/predicoes_classificacao.csv
results/tables/matriz_confusao_melhor_modelo.csv
results/tables/importancia_permutacao.csv
results/tables/metricas_classificacao_sem_vazamento.csv
results/tables/robustez_clusterizacao.csv
results/tables/estabilidade_clusters_ari.csv
results/tables/perfis_clusters_interpretados.csv
results/tables/ablacao_variaveis_classificacao.csv
results/tables/ablacao_variaveis_clusterizacao.csv
```

### Figuras

```text
results/figures/pca_clusters.png
results/figures/heatmap_centroides.png
results/figures/evolucao_clusters_pais.png
results/figures/evolucao_renovaveis_pais.png
results/figures/evolucao_intensidade_carbono_pais.png
results/figures/importancia_permutacao.png
results/figures/matriz_confusao_melhor_modelo.png
```

### Relatórios

```text
results/reports/relatorio_viabilidade.md
results/reports/relatorio_resultados_completo.md
```

### LaTeX

```text
results/latex/tabela_cobertura_paises.tex
results/latex/tabela_metricas_clusterizacao.tex
results/latex/tabela_metricas_classificacao.tex
results/latex/tabela_importancia_variaveis.tex
results/latex/tabela_perfis_clusters.tex
results/latex/tabela_validacao_sem_vazamento.tex
results/latex/tabela_robustez_clusterizacao.tex
```

## Boas práticas de código

- Usar Python claro, modular e comentado em Português Brasil.
- Usar `pathlib.Path` para caminhos.
- Criar diretórios automaticamente com `mkdir(parents=True, exist_ok=True)`.
- Evitar caminhos absolutos fixos.
- Manter compatibilidade com Windows.
- Não alterar nomes de colunas usados por outros scripts sem atualizar todo o pipeline.
- Não remover arquivos existentes sem backup.
- Não incluir credenciais, tokens ou segredos.
- Usar funções pequenas e reutilizáveis.
- Preferir mensagens de erro claras.
- Gerar arquivos CSV com `encoding="utf-8-sig"` quando fizer sentido para abertura no Excel.
- Evitar dependências obrigatórias desnecessárias.
- Tratar bibliotecas opcionais com `try/except ImportError`.

## Backup obrigatório

Antes de modificar arquivos importantes, criar backup em:

```text
_backups/backup_YYYYMMDD_HHMMSS/
```

Copiar arquivos de script para o backup antes da alteração.

## Dependências principais

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- scipy
- joblib
- requests

## Dependências opcionais

- xgboost
- lightgbm
- umap-learn
- shap

Manter o projeto funcionando mesmo sem as dependências opcionais.

## Padrão de interpretação científica

Evitar afirmações exageradas.

Não escrever:

> O modelo comprova a descarbonização dos países.

Preferir:

> Os resultados sugerem padrões distintos de transição energética, com agrupamentos associados à composição da matriz energética, intensidade de carbono e tendência de participação renovável.

Não escrever:

> A regressão logística prevê perfeitamente a transição energética.

Preferir:

> A regressão logística reproduziu com alto desempenho os rótulos derivados da clusterização, indicando que os perfis identificados podem ser separados por combinações lineares das variáveis analisadas.
