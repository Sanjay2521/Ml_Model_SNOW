@echo off
REM Shopping Agent Setup Script for Windows

echo =========================================
echo 🛍️  Shopping Agent Setup
echo =========================================
echo.

REM Check Python
echo Checking Python version...
python --version
if errorlevel 1 (
    echo ✗ Python not found! Please install Python 3.8 or higher
    pause
    exit /b 1
)
echo ✓ Python found
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
echo ✓ Virtual environment created
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo ✓ Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet
echo ✓ Pip upgraded
echo.

REM Install requirements
echo Installing Python packages...
pip install -r requirements.txt --quiet
echo ✓ Python packages installed
echo.

REM Install Playwright browsers
echo Installing Playwright browsers (this may take a few minutes)...
playwright install chromium
echo ✓ Playwright browsers installed
echo.

REM Create .env if it doesn't exist
if not exist .env (
    echo Creating .env file...
    copy .env.example .env
    echo ✓ .env file created
    echo.
    echo ⚠️  IMPORTANT: Edit .env and add your ANTHROPIC_API_KEY
    echo.
) else (
    echo ✓ .env file already exists
    echo.
)

REM Create screenshots directory
if not exist screenshots mkdir screenshots
echo ✓ Screenshots directory created
echo.

echo =========================================
echo ✅ Setup Complete!
echo =========================================
echo.
echo Next steps:
echo 1. Edit .env file and add your Anthropic API key
echo 2. Activate the virtual environment:
echo    venv\Scripts\activate.bat
echo 3. Run the agent:
echo    python main.py --site calvinklein_us --product shirt --size M
echo.
echo Happy shopping! 🛍️🤖
pause
