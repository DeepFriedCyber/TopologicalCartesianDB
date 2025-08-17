# Virtual Environment Setup Script for Benchmarking
# =================================================

Write-Host "🚀 Setting up Virtual Environment for Topological Cartesian Cube Benchmarking" -ForegroundColor Green
Write-Host "=" * 80

$PROJECT_ROOT = "c:\Users\aps33\Projects\topological-cartesian-db"
$VENV_PATH = "$PROJECT_ROOT\venv-benchmarks"
$BENCHMARKS_PATH = "$PROJECT_ROOT\benchmarks"

# Check if virtual environment exists
if (Test-Path $VENV_PATH) {
    Write-Host "✅ Virtual environment already exists at: $VENV_PATH" -ForegroundColor Green
} else {
    Write-Host "🔧 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv $VENV_PATH
    Write-Host "✅ Virtual environment created at: $VENV_PATH" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "🔄 Activating virtual environment..." -ForegroundColor Yellow
& "$VENV_PATH\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "📦 Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install core benchmarking dependencies
Write-Host "📦 Installing core benchmarking dependencies..." -ForegroundColor Yellow
pip install pandas numpy matplotlib seaborn psutil requests plotly scipy tqdm

# Install additional useful packages
Write-Host "📦 Installing additional packages..." -ForegroundColor Yellow
pip install jupyter ipykernel pytest pyyaml jsonschema

# Optional: Install Kaggle API if user wants it
$installKaggle = Read-Host "Do you want to install Kaggle API for real dataset download? (y/N)"
if ($installKaggle -eq "y" -or $installKaggle -eq "Y") {
    Write-Host "📦 Installing Kaggle API..." -ForegroundColor Yellow
    pip install kaggle
    Write-Host "💡 Don't forget to set up your Kaggle API credentials!" -ForegroundColor Cyan
    Write-Host "   1. Go to https://www.kaggle.com/account" -ForegroundColor Cyan
    Write-Host "   2. Click 'Create New API Token'" -ForegroundColor Cyan
    Write-Host "   3. Save kaggle.json to ~/.kaggle/ directory" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "🎉 Virtual environment setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To use the benchmarking system:" -ForegroundColor Cyan
Write-Host "1. Activate the environment: & '$VENV_PATH\Scripts\Activate.ps1'" -ForegroundColor White
Write-Host "2. Navigate to benchmarks: cd '$BENCHMARKS_PATH'" -ForegroundColor White
Write-Host "3. Run benchmarks: python comprehensive_benchmark_suite.py" -ForegroundColor White
Write-Host ""
Write-Host "Quick test commands:" -ForegroundColor Cyan
Write-Host "• Setup data: python setup_kaggle_data.py" -ForegroundColor White
Write-Host "• Run Kaggle benchmark: python kaggle_llm_benchmark.py" -ForegroundColor White
Write-Host "• Run VERSES comparison: python verses_comparison_suite.py" -ForegroundColor White
Write-Host "• Run full suite: python comprehensive_benchmark_suite.py --synthetic-data" -ForegroundColor White
Write-Host ""
Write-Host "=" * 80