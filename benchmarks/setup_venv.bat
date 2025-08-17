@echo off
REM Virtual Environment Setup Script for Benchmarking
REM =================================================

echo 🚀 Setting up Virtual Environment for Topological Cartesian Cube Benchmarking
echo ================================================================================

set PROJECT_ROOT=c:\Users\aps33\Projects\topological-cartesian-db
set VENV_PATH=%PROJECT_ROOT%\venv-benchmarks
set BENCHMARKS_PATH=%PROJECT_ROOT%\benchmarks

REM Check if virtual environment exists
if exist "%VENV_PATH%" (
    echo ✅ Virtual environment already exists at: %VENV_PATH%
) else (
    echo 🔧 Creating virtual environment...
    python -m venv "%VENV_PATH%"
    echo ✅ Virtual environment created at: %VENV_PATH%
)

REM Activate virtual environment
echo 🔄 Activating virtual environment...
call "%VENV_PATH%\Scripts\activate.bat"

REM Upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install core benchmarking dependencies
echo 📦 Installing core benchmarking dependencies...
pip install pandas numpy matplotlib seaborn psutil requests plotly scipy tqdm

REM Install additional useful packages
echo 📦 Installing additional packages...
pip install jupyter ipykernel pytest pyyaml jsonschema

echo.
echo 🎉 Virtual environment setup complete!
echo.
echo To use the benchmarking system:
echo 1. Activate the environment: %VENV_PATH%\Scripts\activate.bat
echo 2. Navigate to benchmarks: cd "%BENCHMARKS_PATH%"
echo 3. Run benchmarks: python comprehensive_benchmark_suite.py
echo.
echo Quick test commands:
echo • Setup data: python setup_kaggle_data.py
echo • Run Kaggle benchmark: python kaggle_llm_benchmark.py
echo • Run VERSES comparison: python verses_comparison_suite.py
echo • Run full suite: python comprehensive_benchmark_suite.py --synthetic-data
echo.
echo ================================================================================

pause