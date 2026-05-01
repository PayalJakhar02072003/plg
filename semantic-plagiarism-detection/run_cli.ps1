# Run backend CLI from repo root (fixes wrong cwd / PYTHONPATH typos).
# Usage: .\run_cli.ps1 "First text." "Second text."
#        .\run_cli.ps1 --file-a a.txt --file-b b.txt

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root
$env:PYTHONPATH = Join-Path $Root "backend"
& python -m plagiarism @args
