@echo off
REM Quick Virtual Environment Activation Script
REM ===========================================

set VENV_PATH=c:\Users\aps33\Projects\topological-cartesian-db\venv-benchmarks

if exist "%VENV_PATH%" (
    echo 🔄 Activating benchmarking virtual environment...
    call "%VENV_PATH%\Scripts\activate.bat"
    echo ✅ Virtual environment activated!
    echo.
    echo You can now run:
    echo • python setup_kaggle_data.py
    echo • python kaggle_llm_benchmark.py
    echo • python comprehensive_benchmark_suite.py
    echo.
) else (
    echo ❌ Virtual environment not found at: %VENV_PATH%
    echo Please run setup_venv.bat first to create the virtual environment.
    pause
)