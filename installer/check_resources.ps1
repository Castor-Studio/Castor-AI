<#
.SYNOPSIS
    Detects local machine capabilities to run CastoStudio AI in real-time
    (gRPC + YOLOv8 + torch) and computes a compatibility score.

.DESCRIPTION
    Native PowerShell script (compatible with Windows 10/11 PowerShell 5.1 and 7+).
    Replaces PyInstaller executables to prevent antivirus false positives.
    Generates an INI file for Inno Setup and a complete JSON report file.

.PARAMETER Ini
    Path to output INI file (read by GetIniString in Inno Setup).

.PARAMETER Out
    Path to output JSON report file.
#>

param(
    [string]$Ini = "",
    [string]$Out = ""
)

$ErrorActionPreference = "SilentlyContinue"

# Recommended vs minimum specs
$CPU_MIN = 4; $CPU_REC = 8
$RAM_MIN = 8.0; $RAM_REC = 16.0
$GPU_VRAM_MIN = 0.0; $GPU_VRAM_REC = 6.0

$WEIGHT_CPU = 0.30
$WEIGHT_RAM = 0.30
$WEIGHT_GPU = 0.40

# 1. CPU Detection
$cpuCores = [System.Environment]::ProcessorCount
if (-not $cpuCores -or $cpuCores -lt 1) {
    try {
        $cpuCores = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
    } catch {
        $cpuCores = 4
    }
}

# 2. RAM Detection (in GB)
$ramGb = 8.0
try {
    $compSys = Get-CimInstance Win32_ComputerSystem
    if ($compSys.TotalPhysicalMemory) {
        $ramGb = [Math]::Round([double]$compSys.TotalPhysicalMemory / 1GB, 1)
    }
} catch {
    $ramGb = 8.0
}

# 3. GPU & VRAM Detection
$gpuName = ""
$gpuVramGb = 0.0

# Try nvidia-smi first (more accurate for NVIDIA VRAM)
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

# Fallback to WMI / CIM if nvidia-smi is unavailable
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
            # Prefer NVIDIA or AMD
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
    $gpuName = "No dedicated GPU detected"
    $gpuVramGb = 0.0
}

# 4. Compute partial scores
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

# 5. Determine tier
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

# 6. Write INI file (for Inno Setup)
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
