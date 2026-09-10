<#
.SYNOPSIS
    Détecte les capacités de la machine locale pour faire tourner CastoStudio AI
    en temps réel (gRPC + YOLOv8 + torch) et calcule un score de compatibilité.

.DESCRIPTION
    Script PowerShell natif (compatible Windows 10/11 PowerShell 5.1 et 7+).
    Remplace les exécutables PyInstaller pour éliminer les faux-positifs antivirus.
    Génère un fichier INI pour Inno Setup et un fichier JSON complet de rapport.

.PARAMETER Ini
    Chemin du fichier INI de sortie (lu par GetIniString dans Inno Setup).

.PARAMETER Out
    Chemin du fichier JSON de sortie de rapport.
#>

param(
    [string]$Ini = "",
    [string]$Out = ""
)

$ErrorActionPreference = "SilentlyContinue"

# Specs recommandées vs minimales
$CPU_MIN = 4; $CPU_REC = 8
$RAM_MIN = 8.0; $RAM_REC = 16.0
$GPU_VRAM_MIN = 0.0; $GPU_VRAM_REC = 6.0

$WEIGHT_CPU = 0.30
$WEIGHT_RAM = 0.30
$WEIGHT_GPU = 0.40

# 1. Détection CPU
$cpuCores = [System.Environment]::ProcessorCount
if (-not $cpuCores -or $cpuCores -lt 1) {
    try {
        $cpuCores = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
    } catch {
        $cpuCores = 4
    }
}

# 2. Détection RAM (en Go)
$ramGb = 8.0
try {
    $compSys = Get-CimInstance Win32_ComputerSystem
    if ($compSys.TotalPhysicalMemory) {
        $ramGb = [Math]::Round([double]$compSys.TotalPhysicalMemory / 1GB, 1)
    }
} catch {
    $ramGb = 8.0
}

# 3. Détection GPU & VRAM
$gpuName = ""
$gpuVramGb = 0.0

# Essayer nvidia-smi en premier (plus précis pour VRAM NVIDIA)
try {
    $smiOut = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null
    if ($smiOut) {
        $firstGpu = ($smiOut -split "`r?`n")[0]
        $parts = $firstGpu -split ","
        if ($parts.Count -ge 2) {
            $gpuName = $parts[0].Trim()
            $vramMb = [double]$parts[1].Trim()
            $gpuVramGb = [Math]::Round($vramMb / 1024.0, 1)
        }
    }
} catch {}

# Fallback WMI / CIM si nvidia-smi absent
if (-not $gpuName) {
    try {
        $gpus = Get-CimInstance Win32_VideoController
        foreach ($g in $gpus) {
            $name = $g.Name
            $adapterRam = $g.AdapterRAM
            $vram = 0.0
            if ($adapterRam -and $adapterRam -gt 0) {
                $vram = [Math]::Round([double]$adapterRam / 1GB, 1)
            }
            # Préférer NVIDIA ou AMD
            if ($name -match "NVIDIA|GeForce|RTX|GTX|Quadro") {
                $gpuName = $name
                $gpuVramGb = [Math]::Max($gpuVramGb, $vram)
                break
            } elseif ($name -match "Radeon|AMD") {
                if (-not $gpuName -or $vram -gt $gpuVramGb) {
                    $gpuName = $name
                    $gpuVramGb = $vram
                }
            } elseif (-not $gpuName) {
                $gpuName = $name
                $gpuVramGb = $vram
            }
        }
    } catch {}
}

if (-not $gpuName) {
    $gpuName = "Aucun GPU dédié détecté"
    $gpuVramGb = 0.0
}

# 4. Calcul des scores partiels
function Calc-SubScore([double]$val, [double]$minVal, [double]$recVal) {
    if ($val -le $minVal) { return 0 }
    if ($val -ge $recVal) { return 100 }
    return [int][Math]::Round(($val -minVal) / ($recVal - $minVal) * 100)
}

$cpuScore = Calc-SubScore $cpuCores $CPU_MIN $CPU_REC
$ramScore = Calc-SubScore $ramGb $RAM_MIN $RAM_REC
$gpuScore = Calc-SubScore $gpuVramGb $GPU_VRAM_MIN $GPU_VRAM_REC

$scorePercent = [int][Math]::Round(
    $cpuScore * $WEIGHT_CPU +
    $ramScore * $WEIGHT_RAM +
    $gpuScore * $WEIGHT_GPU
)
$scorePercent = [Math]::Max(0, [Math]::Min(100, $scorePercent))

# 5. Détermination du palier (Tier)
if ($scorePercent -ge 80) {
    $tier = "excellent"
    $recommended = $true
} elseif ($scorePercent -ge 50) {
    $tier = "ok"
    $recommended = $false
} else {
    $tier = "low"
    $recommended = $false
}

$isoDate = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

# 6. Écriture du fichier INI (pour Inno Setup)
if ($Ini) {
    $iniDir = Split-Path -Parent $Ini
    if ($iniDir -and -not (Test-Path $iniDir)) { New-Item -ItemType Directory -Path $iniDir -Force | Out-Null }
    
    $iniContent = @"
[resources]
checked_at=$isoDate
cpu_cores=$cpuCores
ram_gb=$ramGb
gpu_name=$gpuName
gpu_vram_gb=$gpuVramGb
score_percent=$scorePercent
tier=$tier
recommended=$($recommended.ToString().ToLower())
"@
    Set-Content -Path $Ini -Value $iniContent -Encoding ASCII
}

# 7. Écriture du fichier JSON
$report = [PSCustomObject]@{
    version         = 1
    checked_at      = $isoDate
    os              = "Windows"
    cpu_cores       = $cpuCores
    ram_gb          = $ramGb
    gpu_name        = $gpuName
    gpu_vram_gb     = $gpuVramGb
    cpu_score       = $cpuScore
    ram_score       = $ramScore
    gpu_score       = $gpuScore
    score_percent   = $scorePercent
    tier            = $tier
    recommended     = $recommended
}

$jsonStr = $report | ConvertTo-Json -Depth 3

if ($Out) {
    $outDir = Split-Path -Parent $Out
    if ($outDir -and -not (Test-Path $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }
    Set-Content -Path $Out -Value $jsonStr -Encoding UTF8
}

Write-Output $jsonStr
exit 0
