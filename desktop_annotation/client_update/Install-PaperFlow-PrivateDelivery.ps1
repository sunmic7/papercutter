[CmdletBinding()]
param(
    [string]$InstallRoot = ""
)

$ErrorActionPreference = "Stop"
$packageRoot = Split-Path -Parent $PSScriptRoot
$manifestPath = Join-Path $packageRoot "manifest.json"
if (-not (Test-Path -LiteralPath $manifestPath)) { throw "Update manifest is missing: $manifestPath" }
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json

function Test-PaperFlowInstallRoot([string]$root) {
    if (-not $root) { return $false }
    return (Test-Path -LiteralPath (Join-Path $root "desktop_annotation\PaperFlow-Annotation\resources\app\index.html")) -and
        (Test-Path -LiteralPath (Join-Path $root "desktop_annotation\paperflow_annotation_api.py")) -and
        (Test-Path -LiteralPath (Join-Path $root "review_tools\online_review_sync.py"))
}

function Resolve-PaperFlowInstallRoot([string]$candidate) {
    if (-not $candidate) { return $null }
    $current = [IO.Path]::GetFullPath($candidate)
    for ($level = 0; $level -lt 5; $level++) {
        if (Test-PaperFlowInstallRoot $current) { return $current }
        $parent = Split-Path -Parent $current
        if (-not $parent -or $parent -eq $current) { break }
        $current = $parent
    }
    return $null
}

$resolvedRoot = Resolve-PaperFlowInstallRoot $InstallRoot
if (-not $resolvedRoot) {
    $baseCandidates = @(
        [Environment]::GetFolderPath('Desktop'),
        [Environment]::GetFolderPath('MyDocuments'),
        'C:\PaperFlow', 'D:\PaperFlow', 'E:\PaperFlow', 'F:\PaperFlow',
        'G:\PaperFlow', 'H:\PaperFlow', 'I:\PaperFlow', 'J:\PaperFlow'
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
    foreach ($candidate in $baseCandidates) {
        $resolvedRoot = Resolve-PaperFlowInstallRoot $candidate
        if ($resolvedRoot) { break }
    }
}
if (-not $resolvedRoot) {
    Add-Type -AssemblyName System.Windows.Forms
    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    $dialog.Description = 'Select the PaperFlow installation root (it contains desktop_annotation).'
    if ($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { throw 'Update cancelled.' }
    $resolvedRoot = Resolve-PaperFlowInstallRoot $dialog.SelectedPath
}
if (-not $resolvedRoot) { throw 'The selected folder is not a complete PaperFlow installation.' }

$running = Get-Process -ErrorAction SilentlyContinue | Where-Object {
    try { $_.Path -like '*\PaperFlow-Annotation\*' } catch { $false }
}
if ($running) { throw 'Close PaperFlow completely before updating. No files were changed.' }

$payloadRoot = Join-Path $packageRoot 'payload'
foreach ($entry in @($manifest.files)) {
    $relative = [string]$entry.path
    $source = Join-Path $payloadRoot $relative
    if (-not (Test-Path -LiteralPath $source)) { throw "更新文件缺失：$relative" }
    $actual = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne ([string]$entry.sha256).ToLowerInvariant()) { throw "更新文件校验失败：$relative" }
}

$backupRoot = Join-Path $resolvedRoot ('.paperflow-client-backups\' + [string]$manifest.version)
foreach ($entry in @($manifest.files)) {
    $relative = [string]$entry.path
    $source = Join-Path $payloadRoot $relative
    $target = Join-Path $resolvedRoot $relative
    $backup = Join-Path $backupRoot $relative
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
    if (Test-Path -LiteralPath $target) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $backup) | Out-Null
        Copy-Item -LiteralPath $target -Destination $backup -Force
    }
    Copy-Item -LiteralPath $source -Destination $target -Force
    $actual = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne ([string]$entry.sha256).ToLowerInvariant()) { throw "写入后校验失败：$relative" }
}

Write-Host "Update complete: $($manifest.version)" -ForegroundColor Green
Write-Host '更新完成。重启 PaperFlow 后：审核菜单新增“导入服务器工程”（把员工的题框拉到本机查看/修改）'
Write-Host '和“回传修改到服务器”；最终交付仍然只能在审核员逐页放行之后生成上传。'
