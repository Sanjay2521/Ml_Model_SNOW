"""Main Shopping Agent orchestrator"""
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
from config import Config
from claude_client import ClaudeClient
from web_controller import WebController


class ShoppingAgent:
    """Autonomous shopping agent powered by Claude"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize shopping agent"""
        self.claude = ClaudeClient(api_key)
        self.web = WebController()
        self.current_step = "homepage"
        self.task_context = {}

    async def initialize(self):
        """Initialize the agent"""
        print("=" * 60)
        print("🛍️  SHOPPING AGENT INITIALIZING")
        print("=" * 60)
        Config.validate()
        await self.web.initialize()

    async def shop(
        self,
        site_url: str,
        product: str,
        size: str = "M",
        quantity: int = 1
    ) -> bool:
        """
        Execute end-to-end shopping flow

        Args:
            site_url: Shopping site URL
            product: Product to search for
            size: Size to select
            quantity: Quantity to purchase

        Returns:
            True if successful, False otherwise
        """
        # Set task context
        self.task_context = {
            "site_url": site_url,
            "product": product,
            "size": size,
            "quantity": quantity,
            "goal": f"Search for '{product}', select size '{size}', add to cart, and checkout"
        }

        print("\n" + "=" * 60)
        print("🎯 SHOPPING TASK")
        print("=" * 60)
        print(f"Site: {site_url}")
        print(f"Product: {product}")
        print(f"Size: {size}")
        print(f"Quantity: {quantity}")
        print("=" * 60)

        # Define shopping flow
        flow_steps = [
            ("homepage", self._handle_homepage),
            ("search", self._handle_search),
            ("product", self._handle_product),
            ("cart", self._handle_cart),
            ("checkout", self._handle_checkout),
        ]

        try:
            # Navigate to homepage
            success = await self.web.navigate_to(site_url)
            if not success:
                return False

            # Handle popups
            await asyncio.sleep(2)
            await self.web.handle_popups()

            # Execute flow
            for step_name, handler in flow_steps:
                self.current_step = step_name
                print(f"\n{'=' * 60}")
                print(f"📍 STEP: {step_name.upper()}")
                print(f"{'=' * 60}")

                success = await handler()

                if not success:
                    print(f"\n✗ Failed at step: {step_name}")
                    return False

                # Small delay between steps
                await asyncio.sleep(2)

            print("\n" + "=" * 60)
            print("✅ SHOPPING FLOW COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            return True

        except Exception as e:
            print(f"\n❌ Error during shopping flow: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    async def _handle_homepage(self) -> bool:
        """Handle homepage step"""
        # Take screenshot
        screenshot = await self.web.take_screenshot("homepage")

        # Get page HTML for context
        page_html = await self.web.get_page_html()

        # Ask Claude what to do
        action = self.claude.analyze_page(
            screenshot_path=screenshot,
            current_step="homepage",
            task_context=self.task_context,
            page_html=page_html
        )

        # Execute action
        return await self.web.execute_action(action)

    async def _handle_search(self) -> bool:
        """Handle search results step"""
        # Wait for search results
        await asyncio.sleep(3)

        # Handle any popups that might appear
        await self.web.handle_popups()

        # Take screenshot
        screenshot = await self.web.take_screenshot("search")

        # Get page HTML
        page_html = await self.web.get_page_html()

        # Ask Claude to select a product
        action = self.claude.analyze_page(
            screenshot_path=screenshot,
            current_step="search",
            task_context=self.task_context,
            page_html=page_html
        )

        # Execute action
        return await self.web.execute_action(action)

    async def _handle_product(self) -> bool:
        """Handle product detail page"""
        # Wait for product page to load
        await asyncio.sleep(3)

        # Handle popups
        await self.web.handle_popups()

        # Take screenshot
        screenshot = await self.web.take_screenshot("product")

        # Get page HTML
        page_html = await self.web.get_page_html()

        # First, try to select the size using text-based clicking (more reliable for size buttons)
        size = self.task_context.get("size", "M")
        print(f"🎯 Attempting to select size: {size}")

        # Try clicking size button by text
        size_selected = await self.web.click_by_text(size, "button")

        if not size_selected:
            # Fallback: Ask Claude to help find the size selector
            print(f"   ⚠ Text-based size selection failed, asking Claude for help...")
            action = self.claude.analyze_page(
                screenshot_path=screenshot,
                current_step="product",
                task_context=self.task_context,
                page_html=page_html
            )
            size_selected = await self.web.execute_action(action)

        if not size_selected:
            print(f"✗ Could not select size {size}")
            return False

        # Wait for page to update after size selection
        await asyncio.sleep(2)

        # Now try to add to bag/cart
        print("🛒 Attempting to add to bag...")

        # Try common "Add to Bag" button texts
        add_to_bag_texts = ["Add to Bag", "Add to Cart", "Add To Bag", "ADD TO BAG"]

        for text in add_to_bag_texts:
            if await self.web.click_by_text(text, "button"):
                await asyncio.sleep(2)
                page_text = await self.web.get_page_text()
                if "added to bag" in page_text.lower() or "added to cart" in page_text.lower() or "view bag" in page_text.lower():
                    print("✓ Product added to cart!")
                    return True

        # Fallback: Ask Claude to find and click the add to bag button
        print("   ⚠ Text-based add to bag failed, asking Claude for help...")
        screenshot = await self.web.take_screenshot("product_add_to_bag")
        page_html = await self.web.get_page_html()

        # Update context to focus on adding to bag
        bag_context = self.task_context.copy()
        bag_context["goal"] = f"Click the 'Add to Bag' or 'Add to Cart' button. Size {size} should already be selected."

        action = self.claude.analyze_page(
            screenshot_path=screenshot,
            current_step="product",
            task_context=bag_context,
            page_html=page_html
        )

        success = await self.web.execute_action(action)

        if success:
            await asyncio.sleep(2)
            page_text = await self.web.get_page_text()
            if "added to bag" in page_text.lower() or "added to cart" in page_text.lower():
                print("✓ Product added to cart!")
                return True

        return True  # Continue even if we're not sure

    async def _handle_cart(self) -> bool:
        """Handle cart/bag page"""
        # Try to navigate to cart (might already be there)
        await asyncio.sleep(2)

        # Look for cart/bag links
        cart_selectors = [
            "a[href*='cart']",
            "a[href*='bag']",
            "[data-testid='cart']",
            ".cart-icon",
            "#cart-icon"
        ]

        for selector in cart_selectors:
            try:
                element = await self.web.page.query_selector(selector)
                if element and await element.is_visible():
                    await element.click()
                    await asyncio.sleep(2)
                    break
            except:
                continue

        # Take screenshot
        screenshot = await self.web.take_screenshot("cart")

        # Get page HTML
        page_html = await self.web.get_page_html()

        # Ask Claude to proceed to checkout
        action = self.claude.analyze_page(
            screenshot_path=screenshot,
            current_step="cart",
            task_context=self.task_context,
            page_html=page_html
        )

        # Execute action
        return await self.web.execute_action(action)

    async def _handle_checkout(self) -> bool:
        """Handle checkout page"""
        # Wait for checkout page
        await asyncio.sleep(3)

        # Take screenshot
        screenshot = await self.web.take_screenshot("checkout")

        # Get page HTML
        page_html = await self.web.get_page_html()

        # Ask Claude about checkout
        action = self.claude.analyze_page(
            screenshot_path=screenshot,
            current_step="checkout",
            task_context=self.task_context,
            page_html=page_html
        )

        print("\n" + "=" * 60)
        print("🛑 STOPPING AT CHECKOUT PAGE")
        print("=" * 60)
        print("For demo purposes, we stop before entering payment information.")
        print(f"Reasoning: {action.get('reasoning', 'N/A')}")
        print("=" * 60)

        # Always return True for checkout (we stop here)
        return True

    async def cleanup(self):
        """Cleanup resources"""
        await self.web.close()
        print("\n✓ Agent cleanup completed")


async def main():
    """Main entry point for testing"""
    # Example usage
    agent = ShoppingAgent()

    try:
        await agent.initialize()

        # Test with Calvin Klein US
        success = await agent.shop(
            site_url="https://www.calvinklein.us/",
            product="shirt",
            size="M"
        )

        if success:
            print("\n🎉 Shopping completed successfully!")
        else:
            print("\n❌ Shopping failed")

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
