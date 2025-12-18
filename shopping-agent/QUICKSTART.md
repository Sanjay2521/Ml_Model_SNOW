# 🚀 Quick Start Guide

Get started with the Shopping Agent in 5 minutes!

## ⚡ Quick Setup

### Windows

```bash
# 1. Extract the zip file
# 2. Open Command Prompt in the shopping-agent folder
# 3. Run setup:
setup.bat

# 4. Edit .env file and add your API key:
notepad .env

# 5. Run the agent:
venv\Scripts\activate
python main.py --site calvinklein_us --product "shirt" --size M
```

### Mac/Linux

```bash
# 1. Extract the zip file
unzip shopping-agent.zip
cd shopping-agent

# 2. Run setup:
chmod +x setup.sh
./setup.sh

# 3. Edit .env file and add your API key:
nano .env

# 4. Run the agent:
source venv/bin/activate
python main.py --site calvinklein_us --product "shirt" --size M
```

## 🔑 Get Your API Key

1. Go to: https://console.anthropic.com/
2. Sign up or login
3. Go to "API Keys"
4. Create a new key
5. Copy it to your `.env` file

## 🎯 First Run

Try the demo script for a guided experience:

```bash
python demo.py
```

This will:
- ✓ Navigate to Calvin Klein US
- ✓ Search for a shirt
- ✓ Select size and add to cart
- ✓ Proceed to checkout
- ✓ Save screenshots at each step

## 📸 View Results

After running, check the `screenshots/` folder to see what the agent saw at each step!

## 🆘 Quick Troubleshooting

### "ANTHROPIC_API_KEY is required"
→ Edit `.env` and add your API key

### "Playwright not installed"
→ Run: `playwright install chromium`

### Browser doesn't open
→ Remove `--headless` flag or set `HEADLESS_MODE=False` in `.env`

## 📖 Full Documentation

See `README.md` for complete documentation.

## 🎓 Example Commands

```bash
# Different products
python main.py --site tommy_us --product "jeans" --size "32"
python main.py --site calvinklein_ca --product "jacket" --size "L"

# Custom site
python main.py --url https://www.example.com --product "shoes" --size "10"

# Headless mode (no browser window)
python main.py --site calvinklein_us --product "shirt" --headless
```

## ✨ What's Next?

1. Try different products and sizes
2. Test with different websites
3. Check out the code to understand how it works
4. Modify `config.py` to customize behavior

Happy Shopping! 🛍️🤖
