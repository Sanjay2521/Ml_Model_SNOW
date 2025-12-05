"""Web controller for browser automation using Playwright"""
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any
from playwright.async_api import async_playwright, Browser, Page, TimeoutError as PlaywrightTimeout
from config import Config


class WebController:
    """Controller for web automation using Playwright"""

    def __init__(self):
        """Initialize web controller"""
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.screenshot_counter = 0

    async def initialize(self):
        """Initialize Playwright and browser"""
        self.playwright = await async_playwright().start()

        # Launch browser
        browser_type = getattr(self.playwright, Config.BROWSER_TYPE)
        self.browser = await browser_type.launch(
            headless=Config.HEADLESS_MODE,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox'
            ]
        )

        # Create context with realistic viewport
        context = await self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        # Create page
        self.page = await context.new_page()
        self.page.set_default_timeout(Config.TIMEOUT)

        print("✓ Browser initialized")

    async def navigate_to(self, url: str) -> bool:
        """
        Navigate to URL

        Args:
            url: URL to navigate to

        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"→ Navigating to {url}")
            await self.page.goto(url, wait_until="domcontentloaded")
            await asyncio.sleep(2)  # Wait for page to stabilize
            return True
        except Exception as e:
            print(f"✗ Navigation failed: {str(e)}")
            return False

    async def take_screenshot(self, step_name: str) -> Path:
        """
        Take screenshot of current page

        Args:
            step_name: Name of the current step

        Returns:
            Path to screenshot file
        """
        self.screenshot_counter += 1
        timestamp = int(time.time())
        filename = f"{self.screenshot_counter:02d}_{step_name}_{timestamp}.png"
        screenshot_path = Config.SCREENSHOT_DIR / filename

        await self.page.screenshot(path=str(screenshot_path), full_page=False)
        print(f"📸 Screenshot saved: {screenshot_path}")

        return screenshot_path

    async def execute_action(self, action_data: Dict[str, Any]) -> bool:
        """
        Execute action based on Claude's decision

        Args:
            action_data: Action dictionary from Claude

        Returns:
            True if successful, False otherwise
        """
        action = action_data.get("action")
        selector = action_data.get("selector")
        value = action_data.get("value")
        alternatives = action_data.get("alternatives", [])

        print(f"🤖 Action: {action}")
        print(f"   Reasoning: {action_data.get('reasoning', 'N/A')}")
        print(f"   Confidence: {action_data.get('confidence', 'N/A')}")

        if action == "error":
            print(f"✗ Agent reported error: {action_data.get('reasoning')}")
            return False

        if action == "complete":
            print("✓ Task completed!")
            return True

        if action == "wait":
            wait_time = int(value) if value else 2
            print(f"⏳ Waiting {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            return True

        if action == "scroll":
            return await self._scroll(value)

        # For click and type actions, try selector and alternatives
        selectors_to_try = [selector] + alternatives if selector else alternatives

        for sel in selectors_to_try:
            if not sel:
                continue

            try:
                if action == "click":
                    success = await self._click(sel)
                    if success:
                        return True

                elif action == "type":
                    success = await self._type(sel, value)
                    if success:
                        return True

                elif action == "select":
                    success = await self._select(sel, value)
                    if success:
                        return True

            except Exception as e:
                print(f"   ⚠ Selector '{sel}' failed: {str(e)}")
                continue

        print(f"✗ All selectors failed for action: {action}")
        return False

    async def _click(self, selector: str) -> bool:
        """Click on element"""
        try:
            # Try CSS selector first
            await self.page.wait_for_selector(selector, state="visible", timeout=5000)
            await self.page.click(selector)
            await asyncio.sleep(1.5)
            print(f"   ✓ Clicked: {selector}")
            return True
        except PlaywrightTimeout:
            # Try XPath if CSS failed
            if selector.startswith("//") or selector.startswith("(//"):
                element = await self.page.query_selector(f"xpath={selector}")
                if element:
                    await element.click()
                    await asyncio.sleep(1.5)
                    print(f"   ✓ Clicked (XPath): {selector}")
                    return True
            raise
        except Exception:
            raise

    async def click_by_text(self, text: str, element_type: str = "button") -> bool:
        """
        Click element by visible text using Playwright's text selectors

        Args:
            text: The text to search for
            element_type: Type of element (button, a, etc.)

        Returns:
            True if successful, False otherwise
        """
        strategies = [
            # Try exact text match
            lambda: self.page.get_by_role(element_type, name=text, exact=True),
            # Try partial text match
            lambda: self.page.get_by_role(element_type, name=text, exact=False),
            # Try text selector with element type
            lambda: self.page.locator(f"{element_type}:has-text('{text}')"),
            # Try just text selector
            lambda: self.page.get_by_text(text, exact=True),
            # Try partial text
            lambda: self.page.get_by_text(text, exact=False),
        ]

        for strategy in strategies:
            try:
                element = strategy()
                await element.wait_for(state="visible", timeout=3000)
                await element.click()
                await asyncio.sleep(1.5)
                print(f"   ✓ Clicked element with text: '{text}'")
                return True
            except:
                continue

        return False

    async def _type(self, selector: str, text: str) -> bool:
        """Type text into element"""
        try:
            await self.page.wait_for_selector(selector, state="visible", timeout=5000)
            await self.page.fill(selector, text)
            await asyncio.sleep(0.5)
            # Press Enter after typing (common for search bars)
            await self.page.press(selector, "Enter")
            await asyncio.sleep(2)
            print(f"   ✓ Typed '{text}' into: {selector}")
            return True
        except Exception as e:
            # Try click then type
            await self.page.click(selector)
            await self.page.fill(selector, text)
            await self.page.press(selector, "Enter")
            await asyncio.sleep(2)
            print(f"   ✓ Typed '{text}' into: {selector}")
            return True

    async def _select(self, selector: str, value: str) -> bool:
        """Select option from dropdown"""
        try:
            await self.page.wait_for_selector(selector, state="visible", timeout=5000)
            await self.page.select_option(selector, value)
            await asyncio.sleep(1)
            print(f"   ✓ Selected '{value}' in: {selector}")
            return True
        except:
            # Try clicking on the dropdown and then the option
            await self.page.click(selector)
            await asyncio.sleep(0.5)
            option_selector = f"{selector} option:has-text('{value}')"
            await self.page.click(option_selector)
            await asyncio.sleep(1)
            print(f"   ✓ Selected '{value}' in: {selector}")
            return True

    async def _scroll(self, direction: str) -> bool:
        """Scroll page"""
        try:
            if direction == "down":
                await self.page.evaluate("window.scrollBy(0, 500)")
            elif direction == "up":
                await self.page.evaluate("window.scrollBy(0, -500)")
            elif direction == "bottom":
                await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(1)
            print(f"   ✓ Scrolled: {direction}")
            return True
        except Exception as e:
            print(f"   ✗ Scroll failed: {str(e)}")
            return False

    async def get_page_html(self, max_length: int = 5000) -> str:
        """Get page HTML content"""
        try:
            html = await self.page.content()
            return html[:max_length]
        except:
            return ""

    async def get_page_text(self) -> str:
        """Get visible text from page"""
        try:
            text = await self.page.evaluate("document.body.innerText")
            return text
        except:
            return ""

    async def close(self):
        """Close browser and cleanup"""
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        print("✓ Browser closed")

    async def handle_popups(self):
        """Handle common popups (cookies, newsletters, etc.)"""
        common_popup_selectors = [
            "button:has-text('Accept')",
            "button:has-text('Accept All')",
            "button:has-text('Close')",
            "[aria-label='Close']",
            ".modal-close",
            "#onetrust-accept-btn-handler",
            ".cookie-accept",
            "button:has-text('No Thanks')",
        ]

        for selector in common_popup_selectors:
            try:
                element = await self.page.query_selector(selector)
                if element and await element.is_visible():
                    await element.click()
                    await asyncio.sleep(1)
                    print(f"   ✓ Closed popup: {selector}")
            except:
                continue
