<#
.SYNOPSIS
    Prépare les dépendances requises pour compiler l'installateur Windows Inno Setup.

.DESCRIPTION
    Télécharge l'exécutable uv.exe autonome officiel s'il n'est pas déjà présent.
    Grâce aux scripts PowerShell natifs (check_resources.ps1, write_status.ps1),
    il n'est plus nécessaire d'installer PyInstaller ni de compiler des .exe non signés !
    
    Optionnel : le commutateur -Offline télécharge également le cache des paquets
    Python (PyTorch, Ultralytics, etc.) pour créer un installateur 100% hors-ligne.

.EXAMPLE
    .\installer\build_helpers.ps1
    .\installer\build_helpers.ps1 -Offline
#>

param(
    [switch]$Offline = $false
)

$ErrorActionPreference = "Stop"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  CastoStudio AI — Preparation du build de l'installateur" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$distDir = "installer\dist"
$uvTarget = Join-Path $distDir "uv.exe"

if (-not (Test-Path $distDir)) {
    New-Item -ItemType Directory -Force -Path $distDir | Out-Null
}

if (-not (Test-Path $uvTarget)) {
    Write-Host "--> Telechargement de uv.exe (x86_64 Windows msvc)..." -ForegroundColor Yellow
    $buildDir = "installer\build"
    New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
    $uvZip = Join-Path $buildDir "uv.zip"
    
    $downloadUrl = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
    Invoke-WebRequest -Uri $downloadUrl -OutFile $uvZip
    
    Expand-Archive -Path $uvZip -DestinationPath (Join-Path $buildDir "uv") -Force
    Copy-Item (Join-Path $buildDir "uv\uv.exe") $uvTarget -Force
    Remove-Item -Recurse -Force $buildDir
    Write-Host "--> uv.exe installe dans $uvTarget" -ForegroundColor Green
} else {
    Write-Host "--> uv.exe deja present dans $distDir" -ForegroundColor Green
}

# Mode Offline optionnel (pour les salons, présentations Epitech Experience, etc.)
if ($Offline) {
    Write-Host "--> Mode Offline active : pre-telechargement du cache des roues Python..." -ForegroundColor Yellow
    $cacheDir = "installer\cache"
    New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
    
    & $uvTarget cache clean
    & $uvTarget sync --all-packages --python 3.12 --cache-dir $cacheDir
    Write-Host "--> Cache hors-ligne pret dans $cacheDir" -ForegroundColor Green
}

Write-Host "`nPret ! Vous pouvez maintenant compiler l'installateur :" -ForegroundColor Green
Write-Host "  iscc installer\CastoStudioAI.iss" -ForegroundColor White
Write-Host "Le fichier de sortie sera genere dans installer\output\CastoStudioAI-Setup.exe`n"
