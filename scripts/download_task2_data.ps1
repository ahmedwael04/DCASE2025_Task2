[CmdletBinding()]
param(
    [string]$DataRoot = "data/dcase2025t2"
)

$ErrorActionPreference = "Stop"

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Fetch-And-ExpandZip {
    param(
        [Parameter(Mandatory=$true)][string]$Url,
        [Parameter(Mandatory=$true)][string]$ZipName,
        [Parameter(Mandatory=$true)][string]$DestinationDir
    )

    Ensure-Dir $DestinationDir

    $zipPath = Join-Path $DestinationDir $ZipName

    Write-Host "  ↪ $ZipName"
    if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
        throw "curl.exe not found on PATH. Install curl or use Windows 10+ where curl is included."
    }

    # Use curl for large-file resilience (retry + resume)
    $curlArgs = @(
        "-L",
        "--retry", "10",
        "--retry-all-errors",
        "--retry-delay", "5",
        "--connect-timeout", "30"
    )

    if (Test-Path -LiteralPath $zipPath) {
        $curlArgs += @("-C", "-")
    }

    $curlArgs += @("-o", $zipPath, $Url)
    $maxAttempts = 5
    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
        & curl.exe @curlArgs
        if ($LASTEXITCODE -eq 0) {
            break
        }
        Write-Warning "curl failed (attempt $attempt/$maxAttempts) for $ZipName (exit $LASTEXITCODE). Retrying..."
        Start-Sleep -Seconds 5
    }
    if ($LASTEXITCODE -ne 0) {
        throw "curl download failed for $Url (exit code $LASTEXITCODE)"
    }

    Expand-Archive -Path $zipPath -DestinationPath $DestinationDir -Force
    Remove-Item -LiteralPath $zipPath -Force
}

$projectRoot = (Get-Location).Path

Write-Host "== Development data =="
$devDir = Join-Path $DataRoot "dev_data/raw"
Ensure-Dir $devDir
$devMachines = @("ToyCar","ToyTrain","bearing","fan","gearbox","slider","valve")
foreach ($m in $devMachines) {
    $machineDir = Join-Path $devDir $m
    if (Test-Path -LiteralPath $machineDir) {
        Write-Host "✓ $m exists – skip"
        continue
    }
    $zip = "dev_${m}.zip"
    $url = "https://zenodo.org/records/15097779/files/$zip"
    Fetch-And-ExpandZip -Url $url -ZipName $zip -DestinationDir $devDir
}

Write-Host "== Additional-train & Eval-test data =="
$evalDir = Join-Path $DataRoot "eval_data/raw"
Ensure-Dir $evalDir
$evalMachines = @(
    "AutoTrash","HomeCamera","ToyPet","ToyRCCar",
    "BandSealer","Polisher","ScrewFeeder","CoffeeGrinder"
)
foreach ($m in $evalMachines) {
    $trainDir = Join-Path $evalDir (Join-Path $m "train")
    $testDir  = Join-Path $evalDir (Join-Path $m "test")

    if (-not (Test-Path -LiteralPath $trainDir)) {
        $zip = "eval_data_${m}_train.zip"
        $url = "https://zenodo.org/records/15392814/files/$zip"
        Fetch-And-ExpandZip -Url $url -ZipName $zip -DestinationDir $evalDir
    } else {
        Write-Host "✓ $m/train exists – skip"
    }

    if (-not (Test-Path -LiteralPath $testDir)) {
        $zip = "eval_data_${m}_test.zip"
        $url = "https://zenodo.org/records/15519362/files/$zip"
        Fetch-And-ExpandZip -Url $url -ZipName $zip -DestinationDir $evalDir
    } else {
        Write-Host "✓ $m/test exists – skip"
    }
}

Write-Host "All Task-2 data present ✅"