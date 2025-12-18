"""
Main entry point for Shopping Agent

Usage:
    python main.py --site calvinklein_us --product "shirt" --size M
    python main.py --url https://www.calvinklein.us/ --product "jeans" --size "32"
"""
import asyncio
import argparse
from shopping_agent import ShoppingAgent
from config import Config


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Autonomous Shopping Agent powered by Claude AI"
    )

    parser.add_argument(
        "--site",
        type=str,
        choices=list(Config.SUPPORTED_SITES.keys()),
        help="Predefined site to shop from"
    )

    parser.add_argument(
        "--url",
        type=str,
        help="Custom URL to shop from"
    )

    parser.add_argument(
        "--product",
        type=str,
        required=True,
        help="Product to search for (e.g., 'shirt', 'jeans', 'shoes')"
    )

    parser.add_argument(
        "--size",
        type=str,
        default="M",
        help="Size to select (default: M)"
    )

    parser.add_argument(
        "--quantity",
        type=int,
        default=1,
        help="Quantity to purchase (default: 1)"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode"
    )

    return parser.parse_args()


async def run_shopping_agent(args):
    """Run the shopping agent with given arguments"""

    # Determine URL
    if args.url:
        site_url = args.url
    elif args.site:
        site_url = Config.SUPPORTED_SITES[args.site]
    else:
        print("Error: Either --site or --url must be provided")
        return False

    # Update config if headless mode specified
    if args.headless:
        Config.HEADLESS_MODE = True

    # Create and run agent
    agent = ShoppingAgent()

    try:
        await agent.initialize()

        success = await agent.shop(
            site_url=site_url,
            product=args.product,
            size=args.size,
            quantity=args.quantity
        )

        return success

    finally:
        await agent.cleanup()


def main():
    """Main entry point"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║          🛍️  AUTONOMOUS SHOPPING AGENT 🤖                    ║
║                                                              ║
║          Powered by Claude AI (Anthropic)                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    args = parse_arguments()

    # Run the agent
    success = asyncio.run(run_shopping_agent(args))

    if success:
        print("\n" + "=" * 60)
        print("✅ MISSION ACCOMPLISHED!")
        print("=" * 60)
        print("The agent successfully completed the shopping flow.")
        print("Screenshots have been saved in the 'screenshots' directory.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ MISSION FAILED")
        print("=" * 60)
        print("The agent encountered an error during the shopping flow.")
        print("Check the logs above for more details.")
        print("=" * 60)


if __name__ == "__main__":
    main()
