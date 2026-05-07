param(
    [string]$TestDir = "TEST"
)

Write-Host "Checking ffmpeg availability..."
try {
    & ffmpeg -version > $null 2>&1
} catch {
    Write-Host "ffmpeg not found in PATH. Please install ffmpeg and ensure it's on PATH." -ForegroundColor Red
    exit 1
}

Write-Host "Generating audio from TEST videos..."
python tools\ingest_generated_assets.py --src $TestDir --out tests\fixtures\generated

Write-Host "Running validation runner..."
python tools\validate_test_suite.py

Write-Host "Validation complete."
