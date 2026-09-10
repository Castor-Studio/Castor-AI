<#
.SYNOPSIS
    Écrit le fichier contrat ai_status.json consommé par le front-end CastoStudio.

.DESCRIPTION
    Script PowerShell natif exécuté à la fin de l'installation par Inno Setup.
    Écrit dans %ProgramData%\CastoStudio\ai_status.json
#>

param(
    [string]$Report = "",
    [string]$Confirmed = "false",
    [string]$InstallDir = "",
    [string]$Out = ""
)

$ErrorActionPreference = "Stop"

if (-not $Report -or -not (Test-Path $Report)) {
    Write-Error "Fichier rapport introuvable : $Report"
    exit 1
}

$rawJson = Get-Content -Path $Report -Raw -Encoding UTF8
$rep = $rawJson | ConvertFrom-Json

$isConfirmed = $Confirmed -in @("1", "true", "True", "yes", "oui")
$isRecommended = [bool]$rep.recommended
$aiReady = $isRecommended -or $isConfirmed

$statusObj = [PSCustomObject]@{
    version                 = 1
    checked_at              = $rep.checked_at
    install_dir             = $InstallDir
    cpu_cores               = $rep.cpu_cores
    ram_gb                  = $rep.ram_gb
    gpu_name                = $rep.gpu_name
    gpu_vram_gb             = $rep.gpu_vram_gb
    score_percent           = $rep.score_percent
    tier                    = $rep.tier
    recommended             = $isRecommended
    user_confirmed_override = $isConfirmed
    ai_ready                = $aiReady
}

$targetPath = $Out
if (-not $targetPath) {
    $progData = $env:ProgramData
    if (-not $progData) { $progData = "C:\ProgramData" }
    $targetPath = Join-Path $progData "CastoStudio\ai_status.json"
}

$dir = Split-Path -Parent $targetPath
if (-not (Test-Path $dir)) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
}

$statusJson = $statusObj | ConvertTo-Json -Depth 3
Set-Content -Path $targetPath -Value $statusJson -Encoding UTF8

Write-Host "Statut IA écris dans : $targetPath"
Write-Output $statusJson
exit 0
