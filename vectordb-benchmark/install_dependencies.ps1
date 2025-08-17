# Vector Database Benchmark - Dependency Installation Script

Write-Host "Installing dependencies for Vector Database Benchmark..." -ForegroundColor Green

# Activate virtual environment if it exists, otherwise create it
if (Test-Path ".\venv") {
    Write-Host "Activating existing virtual environment..." -ForegroundColor Yellow
    .\venv\Scripts\Activate.ps1
} else {
    Write-Host "Creating new virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    .\venv\Scripts\Activate.ps1
}

# Install core dependencies
Write-Host "Installing core dependencies..." -ForegroundColor Yellow
pip install numpy pandas matplotlib tqdm requests sentence-transformers

# Install FAISS
Write-Host "Installing FAISS (CPU version)..." -ForegroundColor Yellow
pip install faiss-cpu

# Ask if user wants to install optional dependencies
$installQdrant = Read-Host "Install Qdrant client? (y/n)"
if ($installQdrant -eq "y") {
    Write-Host "Installing Qdrant client..." -ForegroundColor Yellow
    pip install qdrant-client
}

$installMilvus = Read-Host "Install Milvus client? (y/n)"
if ($installMilvus -eq "y") {
    Write-Host "Installing Milvus client..." -ForegroundColor Yellow
    pip install pymilvus
}

# Create directories
Write-Host "Creating necessary directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "benchmark_results"
New-Item -ItemType Directory -Force -Path "sample_data"

Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "To run the benchmark, use: python run_benchmark.py" -ForegroundColor Cyan
