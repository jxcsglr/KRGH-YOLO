param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,
    [ValidateSet("scb", "scsb")]
    [string]$Preset = "scb",
    [string]$Device = "0"
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    python train.py $DatasetRoot --preset $Preset --device $Device @args
}
finally {
    Pop-Location
}
