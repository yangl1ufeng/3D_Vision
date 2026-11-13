$ErrorActionPreference = 'Stop'
$source = $PSScriptRoot
$project = Join-Path $source 'projects\graspgen_feasibility'
$stage = Join-Path $source 'github_upload_ready'
if (Test-Path -LiteralPath $stage) { throw "Upload staging folder already exists: $stage" }
$allowed = @('.py','.md','.json','.txt','.cmd','.ps1','.patch')
function Copy-One([string]$file,[string]$batch) {
  $relative = [IO.Path]::GetRelativePath($source,$file)
  $dest = Join-Path (Join-Path $stage $batch) $relative
  $parent = Split-Path -Parent $dest
  New-Item -ItemType Directory -Path $parent -Force | Out-Null
  Copy-Item -LiteralPath $file -Destination $dest
}
foreach ($name in @('README.md','.gitignore','.gitattributes','organize_workspace.py','prepare_web_upload.ps1')) {
  $path = Join-Path $source $name
  if (Test-Path -LiteralPath $path) { Copy-One $path '01_source' }
}
Get-ChildItem -LiteralPath $project -File | Where-Object { $allowed -contains $_.Extension -and $_.Name -ne 'README.md' } | ForEach-Object { Copy-One $_.FullName '01_source' }
foreach ($name in @('configs','offline_core','simulation')) {
  Get-ChildItem -LiteralPath (Join-Path $project $name) -File -Recurse | Where-Object {
    $allowed -contains $_.Extension -and $_.FullName -notmatch '\\(__pycache__|\.cache)\\'
  } | ForEach-Object { Copy-One $_.FullName '01_source' }
}
Copy-One (Join-Path $project 'README.md') '02_docs_tests'
foreach ($name in @('docs','launchers','setup','tests','experiments')) {
  Get-ChildItem -LiteralPath (Join-Path $project $name) -File -Recurse | Where-Object {
    $allowed -contains $_.Extension -and $_.FullName -notmatch '\\(__pycache__|\.cache)\\'
  } | ForEach-Object { Copy-One $_.FullName '02_docs_tests' }
}
$counts = @('01_source','02_docs_tests') | ForEach-Object {
  $batch = $_
  $files = @(Get-ChildItem -LiteralPath (Join-Path $stage $batch) -File -Recurse)
  [pscustomobject]@{ batch=$batch; files=$files.Count; max_file_MiB=[math]::Round((($files | Measure-Object Length -Maximum).Maximum)/1MB,2) }
}
if (($counts | Measure-Object files -Maximum).Maximum -gt 100) { throw 'A batch exceeds GitHub web upload limit' }
$counts | Format-Table
