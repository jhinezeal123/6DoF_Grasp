#!/usr/bin/env pwsh
# Dong bo code local <-> server qua GitHub (thay cho scp qua Tailscale).
#
#   .\sync.ps1 push "message"   # commit + push local, roi server pull
#   .\sync.ps1 pull             # local pull tu GitHub
#   .\sync.ps1 status           # xem trang thai hai ben
#
# Root local : thu muc chua script nay ($PSScriptRoot = git root)
# Root server: /workspace/6DoF_Grasp/htc      (git root = thu muc lam viec)

param(
    [Parameter(Position = 0)]
    [ValidateSet('push', 'pull', 'status')]
    [string]$Action = 'status',

    [Parameter(Position = 1)]
    [string]$Message = ''
)

$ErrorActionPreference = 'Stop'
# Khong hardcode nua: truoc day gan cung D:\...\htc nen chay tu clone khac
# se am thom thao tac tren clone SAI.
$LocalRoot = $PSScriptRoot
$SrvRoot = '/workspace/6DoF_Grasp/htc'
$SrvHost = 'ktmt'

function Get-SrvHead {
    (ssh -o BatchMode=yes $SrvHost "cd $SrvRoot && git rev-parse --short HEAD").Trim()
}

function Show-Status {
    Push-Location $LocalRoot
    Write-Host '=== LOCAL ===' -ForegroundColor Cyan
    Write-Host "  commit: $(git rev-parse --short HEAD)  branch: $(git branch --show-current)"
    $dirty = git status --short
    if ($dirty) { Write-Host '  thay doi chua commit:' -ForegroundColor Yellow; $dirty | ForEach-Object { "    $_" } }
    else { Write-Host '  sach' -ForegroundColor Green }
    Pop-Location

    Write-Host '=== SERVER ===' -ForegroundColor Cyan
    Write-Host "  commit: $(Get-SrvHead)"
    $sd = ssh -o BatchMode=yes $SrvHost "cd $SrvRoot && git status --short"
    if ($sd) { Write-Host '  thay doi chua commit:' -ForegroundColor Yellow; $sd | ForEach-Object { "    $_" } }
    else { Write-Host '  sach' -ForegroundColor Green }

    Push-Location $LocalRoot
    $l = git rev-parse --short HEAD
    $r = Get-SrvHead
    Write-Host ''
    if ($l -eq $r) { Write-Host "DONG BO ($l)" -ForegroundColor Green }
    else { Write-Host "LECH: local=$l server=$r" -ForegroundColor Red }
    Pop-Location
}

function Invoke-Push {
    if (-not $Message) { throw 'Thieu message. Vi du: .\sync.ps1 push "fix: sua camera"' }
    Push-Location $LocalRoot

    if (-not (git status --short)) {
        Write-Host 'Khong co thay doi de commit.' -ForegroundColor Yellow
    }
    else {
        git add -A
        git commit -q -m $Message
        Write-Host "Da commit: $(git rev-parse --short HEAD)" -ForegroundColor Green
    }

    git push origin main
    Pop-Location
    Write-Host 'Da push len GitHub.' -ForegroundColor Green

    # Server keo ve
    ssh -o BatchMode=yes $SrvHost "cd $SrvRoot && git pull --ff-only origin main" |
        Select-Object -Last 3
    Write-Host "Server da cap nhat: $(Get-SrvHead)" -ForegroundColor Green
}

function Invoke-Pull {
    Push-Location $LocalRoot
    git pull --ff-only origin main
    Write-Host "Local: $(git rev-parse --short HEAD)" -ForegroundColor Green
    Pop-Location
}

switch ($Action) {
    'push' { Invoke-Push }
    'pull' { Invoke-Pull }
    'status' { Show-Status }
}
