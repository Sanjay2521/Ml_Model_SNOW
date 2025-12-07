"""
E-commerce Validation Checks for CK & TH (US & CA)
Comprehensive health checks for shopping flow
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from playwright.async_api import Page


@dataclass
class ValidationResult:
    """Result of a validation check"""
    check_name: str
    status: str  # "PASS", "FAIL", "WARNING", "SKIP"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    checked_by: str = "Shopping Agent"


class EcommerceValidator:
    """Comprehensive validation for e-commerce sites"""

    def __init__(self, page: Page, site_name: str = ""):
        self.page = page
        self.site_name = site_name
        self.results: List[ValidationResult] = []

    async def validate_homepage(self) -> ValidationResult:
        """Validate homepage elements and functionality"""
        print("🔍 Validating Homepage...")

        checks = {
            'logo': False,
            'navigation': False,
            'search': False,
            'cart_icon': False,
            'main_content': False,
            'footer': False
        }

        try:
            logo = await self.page.query_selector('header img, [class*="logo" i]')
            checks['logo'] = logo is not None

            nav = await self.page.query_selector('nav, [role="navigation"]')
            checks['navigation'] = nav is not None

            search = await self.page.query_selector('input[type="search"], input[placeholder*="search" i]')
            checks['search'] = search is not None

            cart = await self.page.query_selector('[class*="cart" i], [aria-label*="cart" i]')
            checks['cart_icon'] = cart is not None

            main = await self.page.query_selector('main, [role="main"], .main-content')
            checks['main_content'] = main is not None

            footer = await self.page.query_selector('footer')
            checks['footer'] = footer is not None

            passed = sum(checks.values())
            total = len(checks)

            status = "PASS" if passed == total else ("WARNING" if passed >= total * 0.7 else "FAIL")
            message = f"Homepage validation: {passed}/{total} checks passed"

            result = ValidationResult(
                check_name="Homepage Validation",
                status=status,
                message=message,
                details=checks
            )

        except Exception as e:
            result = ValidationResult(
                check_name="Homepage Validation",
                status="FAIL",
                message=f"Error: {str(e)}",
                details={}
            )

        self.results.append(result)
        print(f"   {result.status}: {result.message}")
        return result

    async def validate_image_loading(self, stage: str = "homepage") -> ValidationResult:
        """Validate that images are loading correctly"""
        print(f"🔍 Validating Image Loading ({stage})...")

        try:
            images_info = await self.page.evaluate("""
            () => {
                const images = document.querySelectorAll('img');
                const total = images.length;
                let loaded = 0;
                let broken = 0;
                const brokenSrcs = [];

                images.forEach(img => {
                    if (img.complete && img.naturalHeight > 0) {
                        loaded++;
                    } else if (img.complete && img.naturalHeight === 0) {
                        broken++;
                        brokenSrcs.push(img.src);
                    }
                });

                return {
                    total: total,
                    loaded: loaded,
                    broken: broken,
                    brokenSrcs: brokenSrcs.slice(0, 5),
                    loadRate: total > 0 ? (loaded / total * 100).toFixed(2) : 0
                };
            }
            """)

            load_rate = float(images_info['loadRate'])
            status = "PASS" if load_rate >= 95 else ("WARNING" if load_rate >= 80 else "FAIL")
            message = f"{stage}: {images_info['loaded']}/{images_info['total']} images loaded ({load_rate}%)"

            result = ValidationResult(
                check_name=f"Image Loading - {stage}",
                status=status,
                message=message,
                details=images_info
            )

        except Exception as e:
            result = ValidationResult(
                check_name=f"Image Loading - {stage}",
                status="FAIL",
                message=f"Error: {str(e)}",
                details={}
            )

        self.results.append(result)
        print(f"   {result.status}: {result.message}")
        return result

    async def validate_product_variations(self) -> ValidationResult:
        """Validate color and size variation switching in PDP"""
        print("🔍 Validating Product Variations (Color & Size)...")

        try:
            variation_info = await self.page.evaluate("""
            () => {
                const sizeButtons = document.querySelectorAll(
                    '[class*="size" i] button, [data-attr="size"] button'
                );

                const colorSwatches = document.querySelectorAll(
                    '[class*="color" i] button, [class*="swatch" i]'
                );

                return {
                    totalSizes: sizeButtons.length,
                    totalColors: colorSwatches.length,
                    hasSizeVariation: sizeButtons.length > 0,
                    hasColorVariation: colorSwatches.length > 0
                };
            }
            """)

            has_variations = variation_info['hasSizeVariation'] or variation_info['hasColorVariation']
            status = "PASS" if has_variations else "WARNING"
            message = f"Sizes: {variation_info['totalSizes']}, Colors: {variation_info['totalColors']}"

            result = ValidationResult(
                check_name="Product Variations (PDP)",
                status=status,
                message=message,
                details=variation_info
            )

        except Exception as e:
            result = ValidationResult(
                check_name="Product Variations (PDP)",
                status="FAIL",
                message=f"Error: {str(e)}",
                details={}
            )

        self.results.append(result)
        print(f"   {result.status}: {result.message}")
        return result

    async def validate_add_to_cart(self) -> ValidationResult:
        """Validate add to cart functionality"""
        print("🔍 Validating Add to Cart...")

        try:
            cart_info = await self.page.evaluate("""
            () => {
                const addToCartBtn = document.querySelector(
                    '[class*="add-to-cart" i], [class*="addtocart" i], button:has-text("Add to Bag")'
                );

                return {
                    hasButton: !!addToCartBtn,
                    isEnabled: addToCartBtn ? !addToCartBtn.disabled : false,
                    buttonText: addToCartBtn ? addToCartBtn.textContent.trim() : null
                };
            }
            """)

            status = "PASS" if cart_info['hasButton'] and cart_info['isEnabled'] else "WARNING"
            message = f"Add to Cart: {cart_info['buttonText'] or 'Not found'}"

            result = ValidationResult(
                check_name="Add to Cart Button",
                status=status,
                message=message,
                details=cart_info
            )

        except Exception as e:
            result = ValidationResult(
                check_name="Add to Cart Button",
                status="FAIL",
                message=f"Error: {str(e)}",
                details={}
            )

        self.results.append(result)
        print(f"   {result.status}: {result.message}")
        return result

    async def run_full_validation(self, stages: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive validation suite"""
        if stages is None:
            stages = ['current']

        print("\n" + "=" * 60)
        print(f"🔬 RUNNING VALIDATION: {self.site_name}")
        print("=" * 60)

        current_url = self.page.url.lower()

        if 'home' in stages or 'current' in stages:
            if '/' in current_url or 'home' in current_url:
                await self.validate_homepage()

        if 'pdp' in stages or 'current' in stages:
            if 'product' in current_url or '/p/' in current_url:
                await self.validate_image_loading("product_page")
                await self.validate_product_variations()
                await self.validate_add_to_cart()

        return self.generate_report()

    def generate_report(self) -> Dict[str, Any]:
        """Generate validation report"""
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        warnings = sum(1 for r in self.results if r.status == "WARNING")
        total = len(self.results)

        report = {
            "site": self.site_name,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_checks": total,
                "passed": passed,
                "failed": failed,
                "warnings": warnings,
                "pass_rate": f"{(passed/total*100):.2f}%" if total > 0 else "0%"
            },
            "results": [
                {
                    "check": r.check_name,
                    "status": r.status,
                    "message": r.message,
                    "checked_by": r.checked_by
                }
                for r in self.results
            ]
        }

        self._print_report(report)
        return report

    def _print_report(self, report: Dict):
        """Print validation report to console"""
        print("\n" + "=" * 60)
        print("📊 VALIDATION REPORT")
        print("=" * 60)
        print(f"Site: {report['site']}")
        print("\nSUMMARY:")
        print(f"  Total Checks: {report['summary']['total_checks']}")
        print(f"  ✓ Passed: {report['summary']['passed']}")
        print(f"  ✗ Failed: {report['summary']['failed']}")
        print(f"  ⚠ Warnings: {report['summary']['warnings']}")
        print(f"  Pass Rate: {report['summary']['pass_rate']}")
        print("\nDETAILS:")

        for r in report['results']:
            icon = "✓" if r['status'] == "PASS" else ("✗" if r['status'] == "FAIL" else "⚠")
            print(f"  {icon} {r['check']}: {r['message']}")

        print("=" * 60)
