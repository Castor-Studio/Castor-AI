<#
.SYNOPSIS
    Prepares the required dependencies to compile the Windows Inno Setup installer.

.DESCRIPTION
    Downloads the official standalone uv.exe executable if not already present.
    Native PowerShell scripts (check_resources.ps1, write_status.ps1) are used,
    so PyInstaller and unsigned .exe compilation are no longer needed.

    Optional: the -Offline switch pre-downloads the Python package cache
    (PyTorch, Ultralytics, etc.) to create a 100% offline installer.

.EXAMPLE
    .\installer\build_helpers.ps1
    .\installer\build_helpers.ps1 -Offline
#>

param(
    [switch]$Offline = $false
)

$ErrorActionPreference = "Stop"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  CastoStudio AI -- Preparation du build de l'installateur" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { "installer" }
$distDir = Join-Path $scriptDir "dist"
$uvTarget = Join-Path $distDir "uv.exe"

if (-not (Test-Path $distDir)) {
    New-Item -ItemType Directory -Force -Path $distDir | Out-Null
}

if (-not (Test-Path $uvTarget)) {
    Write-Host "--> Telechargement de uv.exe (x86_64 Windows msvc)..." -ForegroundColor Yellow
    $buildDir = Join-Path $scriptDir "build"
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

# Optional Offline mode (for Epitech Experience demos, offline events, etc.)
if ($Offline) {
    Write-Host "--> Mode Offline: pre-telechargement du cache Python..." -ForegroundColor Yellow
    $cacheDir = "installer\cache"
    New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null

    & $uvTarget cache clean
    & $uvTarget sync --all-packages --python 3.12 --cache-dir $cacheDir
    Write-Host "--> Cache hors-ligne pret dans $cacheDir" -ForegroundColor Green
}

Write-Host ""
Write-Host "Pret ! Compilez l'installateur avec :" -ForegroundColor Green
Write-Host "  iscc installer\CastoStudioAI.iss" -ForegroundColor White
Write-Host "Sortie : installer\output\CastoStudioAI-Setup.exe" -ForegroundColor White
Write-Host ""
