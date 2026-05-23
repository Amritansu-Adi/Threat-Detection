# Threat Detection System - Conda Environment Launcher
# Usage: .\run_threat_detect.ps1

Write-Host ""
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host " Threat Detection System - Conda Environment" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

# Initialize conda for PowerShell
Write-Host "Initializing conda..." -ForegroundColor Yellow
& 'C:\ProgramData\anaconda3\shell\condabin\conda-hook.ps1' | Out-Null

# Activate threat-detect environment
Write-Host "Activating 'threat-detect' environment..." -ForegroundColor Yellow
conda activate threat-detect

# Change to project directory
Write-Host "Changing to project directory..." -ForegroundColor Yellow
Set-Location "C:\programing\MachineLearning\person_detect\yolo"

# Run the application
Write-Host ""
Write-Host "Starting Threat Detection System..." -ForegroundColor Green
Write-Host "Python: $(python --version)" -ForegroundColor Green
Write-Host ""

python main.py --verbose

# Keep window open
Read-Host "Press Enter to exit"
