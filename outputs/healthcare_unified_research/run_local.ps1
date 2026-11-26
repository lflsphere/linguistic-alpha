[CmdletBinding()]
param(
    [switch] $PrepareReports
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$python = Join-Path $repo '.venv/Scripts/python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:DO_NOT_TRACK = '1'

function Invoke-ResearchStep {
    param([string] $ScriptName, [string[]] $StepArguments = @())
    Write-Output "Starting $ScriptName"
    & $python (Join-Path $PSScriptRoot $ScriptName) @StepArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$ScriptName failed with exit code $LASTEXITCODE"
    }
}

Push-Location $repo
try {
    Invoke-ResearchStep 'extract_finbert.py' @('--checkpoint-calls', '5', '--batch-size', '32')
    Invoke-ResearchStep 'extract_finbert.py' @('--verify-only')
    Invoke-ResearchStep 'fit_finbert.py'
    Invoke-ResearchStep 'verify_unified.py'
    if ($PrepareReports) {
        Invoke-ResearchStep 'render_unified.py'
    }
    Write-Output 'Local research workflow completed.'
} finally {
    Pop-Location
}
