#!/bin/bash

# Shopping Agent Setup Script

echo "========================================="
echo "🛍️  Shopping Agent Setup"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version found"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
echo "✓ Virtual environment created"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ Pip upgraded"
echo ""

# Install requirements
echo "Installing Python packages..."
pip install -r requirements.txt --quiet
echo "✓ Python packages installed"
echo ""

# Install Playwright browsers
echo "Installing Playwright browsers (this may take a few minutes)..."
playwright install chromium
echo "✓ Playwright browsers installed"
echo ""

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your ANTHROPIC_API_KEY"
    echo ""
else
    echo "✓ .env file already exists"
    echo ""
fi

# Create screenshots directory
mkdir -p screenshots
echo "✓ Screenshots directory created"
echo ""

echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your Anthropic API key"
echo "2. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo "3. Run the agent:"
echo "   python main.py --site calvinklein_us --product shirt --size M"
echo ""
echo "Happy shopping! 🛍️🤖"
