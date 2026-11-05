[CmdletBinding()]
param(
    [string]$ServerAddress = "192.168.1.204",
    [string]$ServerUser = "paperflow-deploy",
    [string]$KeyPath = "",
    [string]$Version = "20261105-review-concurrency"
)

$ErrorActionPreference = "Stop"
$sourceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$templateRoot = $PSScriptRoot
if (-not $KeyPath) { $KeyPath = Join-Path $sourceRoot 'online_review_beta\.paperflow-local-data-server' }
if (-not (Test-Path -LiteralPath $KeyPath)) { throw "Data-server SSH key was not found: $KeyPath" }
$ssh = Join-Path $env:WINDIR 'System32\OpenSSH\ssh.exe'
$scp = Join-Path $env:WINDIR 'System32\OpenSSH\scp.exe'
$target = "${ServerUser}@${ServerAddress}"
$temporaryKey = Join-Path ([IO.Path]::GetTempPath()) ('paperflow-client-publish-' + $PID + '-' + [guid]::NewGuid().ToString('N'))
$currentUser = (& whoami).Trim()
$options = @('-i', $temporaryKey, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8')
$stageParent = Join-Path ([IO.Path]::GetTempPath()) ('PaperFlowClientUpdate-' + [guid]::NewGuid().ToString('N'))
$stage = Join-Path $stageParent $Version

try {
    Copy-Item -LiteralPath $KeyPath -Destination $temporaryKey -Force
    & icacls.exe $temporaryKey /inheritance:r /grant:r "${currentUser}:(R)" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Could not secure the temporary SSH key.' }
    $files = @(
        'desktop_annotation\paperflow_annotation_api.py',
        'desktop_annotation\PaperFlow-Annotation\resources\app\index.html',
        'desktop_annotation\PaperFlow-Annotation\resources\app\logo.png',
        'desktop_annotation\PaperFlow-Annotation\resources\app\assets\paperflow-icon.png',
        'desktop_annotation\PaperFlow-Annotation\resources\app\assets\paperflow-icon.ico',
        'desktop_annotation\PaperFlow-Annotation\resources\app\js\main.js',
        'desktop_annotation\PaperFlow-Annotation\resources\app\js\company-review.js',
        'desktop_annotation\PaperFlow-Annotation\resources\app\js\image-manager.js',
        'desktop_annotation\PaperFlow-Annotation\resources\app\js\utils.js',
        'desktop_annotation\PaperFlow-Annotation\resources\app\js\ui-manager.js',
        'desktop_annotation\PaperFlow-Annotation\resources\app\js\latex-renderer.js',
        'desktop_annotation\PaperFlow-Annotation\resources\app\js\ocr-service.js',
        'desktop_annotation\PaperFlow-Annotation\resources\app\js\employee-workbench.js',
        'desktop_annotation\PaperFlow-Annotation\resources\app\css\employee-workbench.css',
        'desktop_annotation\PaperFlow-Annotation\resources\app\css\components.css',
        'desktop_annotation\PaperFlow-Annotation\resources\app\electron\main.js',
        'desktop_annotation\PaperFlow-Annotation\resources\app\electron\preload.js',
        'review_tools\online_review_sync.py',
        'review_tools\review_session.py',
        'review_tools\manual_annotation_data.py'
    )
    $payload = Join-Path $stage 'payload'
    foreach ($relative in $files) {
        $source = Join-Path $sourceRoot $relative
        if (-not (Test-Path -LiteralPath $source)) { throw "Source file was not found: $relative" }
        $destination = Join-Path $payload $relative
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $destination) | Out-Null
        Copy-Item -LiteralPath $source -Destination $destination -Force
    }
    Copy-Item -LiteralPath (Join-Path $templateRoot 'Install-PaperFlow-PrivateDelivery.ps1') -Destination $stage -Force
    Copy-Item -LiteralPath (Join-Path $templateRoot 'Install-PaperFlow-PrivateDelivery.cmd') -Destination $stage -Force
    Copy-Item -LiteralPath (Join-Path $templateRoot 'README-central-library.md') -Destination (Join-Path $stage 'README.md') -Force
    $manifestFiles = foreach ($relative in $files) {
        [pscustomobject]@{ path = $relative; sha256 = (Get-FileHash -LiteralPath (Join-Path $payload $relative) -Algorithm SHA256).Hash.ToLowerInvariant() }
    }
    [pscustomobject]@{ version = $Version; files = @($manifestFiles) } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $stage 'manifest.json') -Encoding UTF8

    # Keep a reviewable local package as well as the LAN distribution copy.
    $localRelease = Join-Path $sourceRoot ('release\' + $Version)
    New-Item -ItemType Directory -Force -Path $localRelease | Out-Null
    Copy-Item -Path (Join-Path $stage '*') -Destination $localRelease -Recurse -Force

    & $ssh @options $target 'cmd.exe /d /c "if not exist J:\PaperFlowData\Public\System\ClientTools\PaperFlowClientUpdates mkdir J:\PaperFlowData\Public\System\ClientTools\PaperFlowClientUpdates"'
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the client-update directory on the server.' }
    & $scp @options -r $stage "${target}:J:/PaperFlowData/Public/System/ClientTools/PaperFlowClientUpdates/"
    if ($LASTEXITCODE -ne 0) { throw 'Client update package upload failed.' }
    Write-Host "Published: \\$ServerAddress\PaperFlowTest\Public\System\ClientTools\PaperFlowClientUpdates\$Version" -ForegroundColor Green
    Write-Host "Local package: $localRelease" -ForegroundColor Green
} finally {
    $resolvedStageParent = [IO.Path]::GetFullPath($stageParent)
    $resolvedTemp = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if (-not $resolvedStageParent.StartsWith($resolvedTemp, [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid package staging cleanup target.' }
    if (Test-Path -LiteralPath $resolvedStageParent) { Remove-Item -LiteralPath $resolvedStageParent -Recurse -Force }
    if (Test-Path -LiteralPath $temporaryKey) {
        # Copying a key can preserve a different owner. Grant the invoking user
        # full control only after SSH has finished, so the disposable key is
        # still safely removed even in a constrained desktop session.
        & icacls.exe $temporaryKey /grant:r "${currentUser}:(F)" | Out-Null
        Remove-Item -LiteralPath $temporaryKey -Force
    }
}
