<#
.SYNOPSIS
    Organiza a raiz do projeto MERCOSUL Descarbonizacao ML.

.DESCRIPTION
    Execucoes antigas do pipeline geraram copias de tabelas (.csv), tabelas
    LaTeX (.tex), figuras (.png) e relatorios (.md) tanto na raiz do projeto
    quanto em results/. Este script mantem APENAS as copias de results/,
    movendo (nao apagando) as duplicatas da raiz para uma pasta de backup.

    Seguranca:
      - So move um arquivo da raiz se um arquivo de MESMO NOME ja existir
        em algum lugar de results/ (a copia canonica e preservada).
      - Nada e apagado: tudo vai para _backups\raiz_limpeza_<timestamp>\.
      - README.md, requirements*.txt, .gitignore e este .ps1 nunca sao tocados.

.NOTES
    Execute a partir da raiz do projeto:  .\organizar_resultados.ps1
#>

[CmdletBinding()]
param(
    [switch]$WhatIfOnly  # use -WhatIfOnly para apenas simular
)

$ErrorActionPreference = "Stop"

# Raiz = pasta onde este script esta
$root      = $PSScriptRoot
$resultsDir = Join-Path $root "results"

if (-not (Test-Path $resultsDir)) {
    Write-Host "[ERRO] Pasta results/ nao encontrada. Abortando." -ForegroundColor Red
    exit 1
}

# Pasta de backup com timestamp
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backupDir = Join-Path $root ("_backups\raiz_limpeza_{0}" -f $timestamp)

# Arquivos a NUNCA mover
$protegidos = @("README.md", "organizar_resultados.ps1")

# Extensoes geradas pelo pipeline que podem ter "vazado" para a raiz
$extensoes = @("*.csv", "*.tex", "*.png", "*.md")

# Indexa todos os nomes de arquivo existentes em results/ (busca recursiva)
$nomesEmResults = @{}
Get-ChildItem -Path $resultsDir -Recurse -File | ForEach-Object {
    $nomesEmResults[$_.Name] = $_.FullName
}

# Coleta candidatos na RAIZ (sem recursao)
$candidatos = Get-ChildItem -Path $root -File -Include $extensoes |
    Where-Object { $protegidos -notcontains $_.Name }

$movidos = @()
$ignorados = @()

foreach ($arq in $candidatos) {
    if ($nomesEmResults.ContainsKey($arq.Name)) {
        # Duplicata confirmada -> mover para backup
        if (-not (Test-Path $backupDir)) {
            New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
        }
        $destino = Join-Path $backupDir $arq.Name
        if ($WhatIfOnly) {
            Write-Host ("[SIMULACAO] moveria: {0}" -f $arq.Name) -ForegroundColor Yellow
        } else {
            Move-Item -Path $arq.FullName -Destination $destino -Force
        }
        $movidos += $arq.Name
    } else {
        # Sem copia em results/ -> nao mexer
        $ignorados += $arq.Name
    }
}

Write-Host ""
Write-Host "==================== RESUMO ====================" -ForegroundColor Cyan
Write-Host ("Duplicatas {0}: {1}" -f ($(if($WhatIfOnly){"a mover (simulacao)"}else{"movidas para backup"}), $movidos.Count)) -ForegroundColor Green
if (-not $WhatIfOnly -and $movidos.Count -gt 0) {
    Write-Host ("Backup em: {0}" -f $backupDir) -ForegroundColor Green
}
if ($ignorados.Count -gt 0) {
    Write-Host ("Arquivos da raiz SEM copia em results/ (mantidos): {0}" -f $ignorados.Count) -ForegroundColor DarkYellow
    $ignorados | ForEach-Object { Write-Host ("   - {0}" -f $_) -ForegroundColor DarkYellow }
}
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "OK: a raiz esta organizada. Todos os resultados permanecem em results/." -ForegroundColor Green
