param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$InstallerArgs
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$installPy = Join-Path $scriptDir "install.py"

if ($env:PYTHON_BIN) {
  & $env:PYTHON_BIN $installPy @InstallerArgs
  exit $LASTEXITCODE
}

if (Get-Command py -ErrorAction SilentlyContinue) {
  & py -3 $installPy @InstallerArgs
  exit $LASTEXITCODE
}

if (Get-Command python -ErrorAction SilentlyContinue) {
  & python $installPy @InstallerArgs
  exit $LASTEXITCODE
}

Write-Error "Python 3.10+ is required but was not found in PATH."
exit 1
