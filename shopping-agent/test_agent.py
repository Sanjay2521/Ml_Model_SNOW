"""
Test script for shopping agent

Run this to verify the agent is working correctly.
"""
import asyncio
import os
from pathlib import Path


async def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from config import Config
        from claude_client import ClaudeClient
        from web_controller import WebController
        from shopping_agent import ShoppingAgent
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {str(e)}")
        return False


async def test_config():
    """Test configuration"""
    print("\nTesting configuration...")
    try:
        from config import Config

        print(f"  - Claude Model: {Config.CLAUDE_MODEL}")
        print(f"  - Browser: {Config.BROWSER_TYPE}")
        print(f"  - Headless: {Config.HEADLESS_MODE}")
        print(f"  - Timeout: {Config.TIMEOUT}ms")
        print(f"  - Screenshot Dir: {Config.SCREENSHOT_DIR}")
        print(f"  - Supported Sites: {len(Config.SUPPORTED_SITES)}")

        # Check API key
        if Config.ANTHROPIC_API_KEY:
            api_key_preview = Config.ANTHROPIC_API_KEY[:10] + "..." + Config.ANTHROPIC_API_KEY[-4:]
            print(f"  - API Key: {api_key_preview} ✓")
        else:
            print(f"  - API Key: NOT SET ✗")
            print("\n⚠️  Please set ANTHROPIC_API_KEY in .env file")
            return False

        print("✓ Configuration valid")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {str(e)}")
        return False


async def test_dependencies():
    """Test that required dependencies are installed"""
    print("\nTesting dependencies...")
    required = [
        ("anthropic", "Anthropic API client"),
        ("playwright", "Web automation"),
        ("dotenv", "Environment variables"),
        ("PIL", "Image processing"),
    ]

    all_ok = True
    for package, description in required:
        try:
            if package == "PIL":
                __import__("PIL")
            elif package == "dotenv":
                __import__("dotenv")
            else:
                __import__(package)
            print(f"  ✓ {package:15s} - {description}")
        except ImportError:
            print(f"  ✗ {package:15s} - {description} (NOT INSTALLED)")
            all_ok = False

    if all_ok:
        print("✓ All dependencies installed")
    else:
        print("✗ Some dependencies missing. Run: pip install -r requirements.txt")

    return all_ok


async def test_playwright():
    """Test Playwright installation"""
    print("\nTesting Playwright browsers...")
    try:
        from playwright.async_api import async_playwright

        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://example.com")
        title = await page.title()
        await browser.close()
        await playwright.stop()

        print(f"  ✓ Chromium browser working (loaded: {title})")
        return True
    except Exception as e:
        print(f"  ✗ Playwright test failed: {str(e)}")
        print("  Run: playwright install chromium")
        return False


async def test_screenshots_dir():
    """Test screenshots directory"""
    print("\nTesting screenshots directory...")
    try:
        from config import Config

        if Config.SCREENSHOT_DIR.exists():
            print(f"  ✓ Screenshots directory exists: {Config.SCREENSHOT_DIR}")
            return True
        else:
            Config.SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created screenshots directory: {Config.SCREENSHOT_DIR}")
            return True
    except Exception as e:
        print(f"  ✗ Screenshots directory test failed: {str(e)}")
        return False


async def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 SHOPPING AGENT - SYSTEM TEST")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Dependencies", test_dependencies),
        ("Screenshots Directory", test_screenshots_dir),
        ("Playwright", test_playwright),
    ]

    results = []
    for test_name, test_func in tests:
        result = await test_func()
        results.append((test_name, result))

    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)

    all_passed = True
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:30s} {status}")
        if not result:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✅ ALL TESTS PASSED!")
        print("\nYou're ready to run the shopping agent:")
        print("  python main.py --site calvinklein_us --product shirt --size M")
        print("\nOr try the demo:")
        print("  python demo.py")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nPlease fix the issues above before running the agent.")
        print("\nCommon fixes:")
        print("  - Set ANTHROPIC_API_KEY in .env file")
        print("  - Run: pip install -r requirements.txt")
        print("  - Run: playwright install chromium")

    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
