# 🛍️ Autonomous Shopping Agent

An intelligent shopping agent powered by Claude AI (Anthropic) that can autonomously navigate e-commerce websites, search for products, select sizes, add items to cart, and proceed through checkout.

## 🌟 Features

- **Autonomous Navigation**: Uses Claude AI's vision capabilities to understand and navigate websites
- **Screenshot Analysis**: Captures and analyzes webpage screenshots to make intelligent decisions
- **End-to-End Flow**: Complete shopping flow from homepage to checkout
- **Multi-Site Support**: Works with Calvin Klein, Tommy Hilfiger, and other e-commerce sites
- **Intelligent Error Handling**: Retries failed actions and tries alternative selectors
- **Visual Feedback**: Saves screenshots at each step for debugging and verification

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Shopping Agent                           │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Claude     │  │     Web      │  │  Shopping    │     │
│  │   Client     │◄─┤  Controller  │◄─┤    Agent     │     │
│  │  (AI Brain)  │  │  (Playwright)│  │ (Orchestrator│     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                  │                  │             │
│         └──────────────────┴──────────────────┘             │
│                Vision + Action + Control                     │
└─────────────────────────────────────────────────────────────┘
```

### Components

1. **Claude Client** (`claude_client.py`):
   - Interfaces with Anthropic's Claude API
   - Analyzes screenshots using vision capabilities
   - Makes intelligent decisions about next actions
   - Returns structured actions (click, type, select, etc.)

2. **Web Controller** (`web_controller.py`):
   - Manages Playwright browser automation
   - Executes actions determined by Claude
   - Handles screenshots and page interactions
   - Manages popups and common UI elements

3. **Shopping Agent** (`shopping_agent.py`):
   - Orchestrates the entire shopping flow
   - Manages state transitions between steps
   - Coordinates Claude and Web Controller
   - Implements shopping logic (homepage → search → product → cart → checkout)

## 📋 Prerequisites

- Python 3.8 or higher
- Anthropic API key ([Get one here](https://console.anthropic.com/))
- Internet connection

## 🚀 Installation

### Step 1: Extract the Package

```bash
unzip shopping-agent.zip
cd shopping-agent
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Playwright Browsers

```bash
playwright install chromium
```

### Step 5: Configure API Key

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Anthropic API key
# ANTHROPIC_API_KEY=your_actual_api_key_here
```

On Windows, you can edit `.env` with Notepad:
```bash
notepad .env
```

On Mac/Linux:
```bash
nano .env
```

## 💻 Usage

### Basic Usage

```bash
# Shop for a shirt on Calvin Klein US
python main.py --site calvinklein_us --product "shirt" --size M

# Shop for jeans on Tommy Hilfiger US
python main.py --site tommy_us --product "jeans" --size "32"

# Use custom URL
python main.py --url https://www.calvinklein.ca/ --product "jacket" --size L
```

### Advanced Usage

```bash
# Run in headless mode (no browser window)
python main.py --site calvinklein_us --product "shirt" --headless

# Specify quantity
python main.py --site tommy_us --product "shoes" --size "10" --quantity 2
```

### Supported Sites

The agent comes pre-configured with these sites:

- `calvinklein_us` - https://www.calvinklein.us/
- `calvinklein_ca` - https://www.calvinklein.ca/
- `tommy_us` - https://usa.tommy.com/
- `tommy_ca` - https://ca.tommy.com/en

You can also use `--url` to test with any other e-commerce site.

## 📸 Screenshots

The agent automatically captures screenshots at each step:

```
screenshots/
├── 01_homepage_1638360000.png
├── 02_search_1638360010.png
├── 03_product_1638360020.png
├── 04_cart_1638360030.png
└── 05_checkout_1638360040.png
```

These are saved in the `screenshots/` directory for debugging and verification.

## 🔧 Configuration

Edit `config.py` or `.env` to customize:

```python
# Browser settings
HEADLESS_MODE=False          # Set to True for headless mode
BROWSER_TYPE=chromium        # chromium, firefox, or webkit
TIMEOUT=30000                # Timeout in milliseconds

# Agent settings
MAX_RETRIES=3                # Max retries for failed actions
AGENT_TEMPERATURE=0.7        # Claude temperature (0-1)
MAX_TOKENS=4096              # Max tokens for Claude response
```

## 🎯 How It Works

### Shopping Flow

```
1. Homepage
   ├─ Navigate to site
   ├─ Close popups/cookies
   └─ Find and focus search bar

2. Search
   ├─ Enter product search term
   ├─ Wait for results
   └─ Select first relevant product

3. Product Page
   ├─ Analyze product details
   ├─ Select requested size
   └─ Click "Add to Cart/Bag"

4. Cart
   ├─ Verify product in cart
   └─ Proceed to checkout

5. Checkout
   └─ Stop (demo mode - no payment)
```

### AI Decision Making

At each step, the agent:

1. **Captures Screenshot**: Takes a full-page screenshot
2. **Analyzes with Claude**: Sends screenshot to Claude API
3. **Gets Action**: Claude analyzes the page and returns action
4. **Executes Action**: Web controller executes the action
5. **Validates**: Checks if action was successful
6. **Repeats**: Moves to next step or retries

### Example Claude Response

```json
{
  "action": "type",
  "selector": "input[name='search']",
  "value": "shirt",
  "reasoning": "Found search input in header, will type product name",
  "confidence": "high",
  "alternatives": ["#search-input", "[placeholder*='Search']"]
}
```

## 🐛 Troubleshooting

### "ANTHROPIC_API_KEY is required"

Make sure you've created a `.env` file and added your API key:

```bash
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=your_key_here
```

### "Playwright not installed"

Run:
```bash
playwright install chromium
```

### "Navigation failed" or "Timeout"

1. Check your internet connection
2. Try increasing timeout in `.env`: `TIMEOUT=60000`
3. The site might be blocking automation - try with `HEADLESS_MODE=False`

### Agent makes wrong decisions

1. Check screenshots to see what the agent saw
2. Adjust `AGENT_TEMPERATURE` (lower = more conservative)
3. Some sites have complex layouts that may confuse the agent
4. Try a different site or product

### Browser doesn't close

Press Ctrl+C to force stop, then run:
```bash
pkill -f chromium
```

## 🔒 Security & Privacy

- **No Payment Processing**: Agent stops at checkout page
- **No Data Storage**: No personal information is stored
- **API Key Security**: Keep your `.env` file private
- **Local Execution**: Everything runs on your machine

## 📝 API Key Setup

1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key to your `.env` file

## 🎓 Use Cases

- **E-commerce Testing**: Automated testing of shopping flows
- **Price Monitoring**: Track product availability and prices
- **Research**: Study e-commerce UX patterns
- **Education**: Learn about AI agents and web automation
- **Accessibility**: Help users with disabilities shop online

## ⚠️ Important Notes

1. **Rate Limiting**: Be respectful of websites - don't run too frequently
2. **Terms of Service**: Ensure your use complies with site ToS
3. **Demo Purpose**: This is for demonstration - don't use for actual purchases
4. **Site Changes**: Websites change frequently, agent may need adjustments
5. **API Costs**: Claude API calls cost money - monitor your usage

## 🚧 Limitations

- Some sites have strong anti-bot measures
- Complex CAPTCHA will block the agent
- Dynamic SPAs may be challenging
- Checkout forms are not filled (by design)
- Success rate varies by site complexity

## 🔮 Future Enhancements

- Support for more shopping sites
- CAPTCHA solving integration
- Product comparison features
- Price tracking and alerts
- Multi-product shopping carts
- Form filling for checkout (with user consent)

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Support

For issues or questions:
1. Check the screenshots to see what went wrong
2. Review the console output for error messages
3. Try with a different site or product
4. Adjust configuration settings

## 🙏 Acknowledgments

- **Anthropic** for the Claude AI API
- **Playwright** for excellent web automation
- **Python** community for amazing libraries

---

**Built with ❤️ using Claude AI**

Happy Shopping! 🛍️🤖
