"""
Intelligent Size Selector for Shopping Agents
Uses semantic analysis and multiple fallback strategies
"""

import asyncio
from playwright.async_api import Page
from typing import Optional, Dict, Any


async def select_size_intelligently(page: Page, size: str) -> bool:
    """
    Intelligently select a size on any e-commerce site.

    Args:
        page: Playwright page object
        size: Size to select (e.g., "M", "L", "XL")

    Returns:
        True if successful, False otherwise
    """
    size = size.upper().strip()

    strategies = [
        ("aria_role", lambda: _try_aria_role(page, size)),
        ("text_content", lambda: _try_text_content(page, size)),
        ("data_attributes", lambda: _try_data_attributes(page, size)),
        ("javascript", lambda: _try_javascript(page, size)),
    ]

    for strategy_name, strategy_func in strategies:
        try:
            result = await strategy_func()
            if result:
                print(f"   ✓ Size selected using: {strategy_name}")
                return True
        except Exception as e:
            continue

    return False


async def _try_aria_role(page: Page, size: str) -> bool:
    """Try using ARIA roles"""
    try:
        locator = page.get_by_role("button", name=size, exact=True)
        if await locator.count() > 0:
            await locator.first.click(timeout=3000)
            return True
    except:
        pass
    return False


async def _try_text_content(page: Page, size: str) -> bool:
    """Try using text content"""
    try:
        await page.get_by_text(size, exact=True).first.click(timeout=3000, force=True)
        return True
    except:
        pass
    return False


async def _try_data_attributes(page: Page, size: str) -> bool:
    """Try using data attributes"""
    selectors = [
        f'[data-attr-value="{size}"]',
        f'[data-attr-value="{size.lower()}"]',
        f'[data-size="{size}"]',
        f'[data-size="{size.lower()}"]',
    ]

    for selector in selectors:
        try:
            await page.click(selector, timeout=2000, force=True)
            return True
        except:
            continue
    return False


async def _try_javascript(page: Page, size: str) -> bool:
    """Try using JavaScript to click"""
    js_code = """
    (targetSize) => {
        const sizeVariants = [targetSize, targetSize.toLowerCase(), targetSize.toUpperCase()];
        const buttons = document.querySelectorAll('button, [role="button"], label');

        for (const btn of buttons) {
            const text = btn.textContent?.trim() || '';
            if (sizeVariants.includes(text)) {
                btn.click();
                return true;
            }
        }
        return false;
    }
    """

    try:
        result = await page.evaluate(js_code, size)
        if result:
            await asyncio.sleep(0.5)
            return True
    except:
        pass
    return False
