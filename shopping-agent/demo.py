"""
Demo script to showcase the shopping agent

This script runs a simple demo without requiring command line arguments.
"""
import asyncio
from shopping_agent import ShoppingAgent
from config import Config


async def run_demo():
    """Run a demonstration of the shopping agent"""

    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║          🛍️  SHOPPING AGENT DEMO 🤖                         ║
║                                                              ║
║          This demo will show the agent navigating            ║
║          Calvin Klein US and searching for a shirt.          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    print("\n⚙️  Configuration:")
    print(f"   - Site: Calvin Klein US")
    print(f"   - Product: Men's Shirt")
    print(f"   - Size: M")
    print(f"   - Headless Mode: {Config.HEADLESS_MODE}")
    print(f"   - Claude Model: {Config.CLAUDE_MODEL}")

    input("\nPress Enter to start the demo...")

    # Create agent
    agent = ShoppingAgent()

    try:
        # Initialize
        await agent.initialize()

        # Run shopping flow
        success = await agent.shop(
            site_url="https://www.calvinklein.us/",
            product="men's shirt",
            size="M"
        )

        if success:
            print("\n" + "=" * 60)
            print("🎉 DEMO COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print("\nThe agent demonstrated the complete shopping flow:")
            print("1. ✓ Navigated to homepage")
            print("2. ✓ Searched for product")
            print("3. ✓ Selected a product")
            print("4. ✓ Chose size and added to cart")
            print("5. ✓ Proceeded to checkout")
            print("\nCheck the 'screenshots' folder to see what the agent saw!")
            print("=" * 60)
        else:
            print("\n❌ Demo encountered an issue. Check logs above.")

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {str(e)}")
    finally:
        await agent.cleanup()


def main():
    """Main entry point"""
    try:
        # Check if API key is set
        Config.validate()
    except ValueError as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n📝 Please follow these steps:")
        print("1. Copy .env.example to .env")
        print("2. Edit .env and add your ANTHROPIC_API_KEY")
        print("3. Run this demo again")
        return

    # Run the demo
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
