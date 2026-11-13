$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$projectRoot = $PSScriptRoot
$repoRoot = Split-Path (Split-Path $projectRoot)
$commit = '37f7d66d3196490618bf89ffe2b49dcb7e07aec3'
$cache = Join-Path $repoRoot '.downloads\robot-model-inspection'
$destination = Join-Path $projectRoot 'third_party'
New-Item -ItemType Directory -Force $cache,$destination | Out-Null
$archive = Join-Path $cache ('ur5-robotiq-' + $commit + '.zip')
$model = Join-Path $destination ('pybullet_ur5_robotiq-' + $commit)
if (-not (Test-Path -LiteralPath $model)) {
    Invoke-WebRequest -Uri ('https://codeload.github.com/ElectronicElephant/pybullet_ur5_robotiq/zip/' + $commit) -OutFile $archive -TimeoutSec 180
    Expand-Archive -LiteralPath $archive -DestinationPath $destination
}
if (-not (Test-Path -LiteralPath (Join-Path $model 'urdf\ur5_robotiq_140.urdf'))) {
    throw 'Model download is incomplete; the URDF is missing.'
}
Write-Output ('Model source: ' + $model)
