"""Claude API client for agentic shopping behavior"""
import base64
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import anthropic
from config import Config


class ClaudeClient:
    """Client for interacting with Claude API for agentic behavior"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Claude client"""
        self.api_key = api_key or Config.ANTHROPIC_API_KEY
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.conversation_history = []

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.standard_b64encode(image_file.read()).decode("utf-8")

    def analyze_page(
        self,
        screenshot_path: Path,
        current_step: str,
        task_context: Dict[str, Any],
        page_html: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze webpage screenshot and determine next action

        Args:
            screenshot_path: Path to screenshot
            current_step: Current step in the shopping flow
            task_context: Context about the shopping task
            page_html: Optional HTML content for text analysis

        Returns:
            Dictionary with action, selector, and value
        """
        # Encode screenshot
        image_data = self._encode_image(screenshot_path)

        # Build system prompt
        system_prompt = self._build_system_prompt(current_step, task_context)

        # Build user prompt
        user_prompt = self._build_user_prompt(current_step, task_context, page_html)

        # Create messages with image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]

        # Add conversation history
        if self.conversation_history:
            messages = self.conversation_history + messages

        # Call Claude API
        response = self.client.messages.create(
            model=Config.CLAUDE_MODEL,
            max_tokens=Config.MAX_TOKENS,
            temperature=Config.AGENT_TEMPERATURE,
            system=system_prompt,
            messages=messages
        )

        # Parse response
        action_data = self._parse_response(response.content[0].text)

        # Update conversation history
        self.conversation_history = messages + [
            {
                "role": "assistant",
                "content": response.content[0].text
            }
        ]

        # Keep only last 4 exchanges to manage context
        if len(self.conversation_history) > 8:
            self.conversation_history = self.conversation_history[-8:]

        return action_data

    def _build_system_prompt(self, current_step: str, task_context: Dict[str, Any]) -> str:
        """Build system prompt for Claude"""
        return f"""You are an expert shopping agent that can navigate e-commerce websites autonomously.

Your task is to help complete online shopping by analyzing webpage screenshots and determining the next action.

Current Step: {current_step}
Shopping Goal: {task_context.get('goal', 'Complete shopping task')}

IMPORTANT INSTRUCTIONS:
1. Analyze the screenshot carefully to understand the page layout and elements
2. Identify the appropriate action based on the current step
3. Provide specific CSS selectors or XPath for interactive elements
4. For text input, provide the exact text to enter
5. Be precise and accurate - a wrong action could break the flow
6. If you cannot find the required element, suggest an alternative or indicate failure

RESPONSE FORMAT (JSON):
{{
    "action": "click|type|scroll|select|wait|complete|error",
    "selector": "CSS selector or XPath",
    "value": "text value for type/select actions",
    "reasoning": "brief explanation of why this action",
    "confidence": "high|medium|low",
    "alternatives": ["alternative selectors if primary fails"]
}}

ACTIONS:
- click: Click on an element (button, link, etc.)
- type: Enter text in input field
- scroll: Scroll page (value: up/down/bottom)
- select: Select dropdown option
- wait: Wait for page to load
- complete: Task completed successfully
- error: Cannot proceed, explain in reasoning
"""

    def _build_user_prompt(
        self,
        current_step: str,
        task_context: Dict[str, Any],
        page_html: Optional[str] = None
    ) -> str:
        """Build user prompt based on current step"""

        product = task_context.get('product', 'shirt')
        size = task_context.get('size', 'M')

        prompts = {
            "homepage": f"""This is the homepage. I need to search for a '{product}'.

Please analyze the screenshot and tell me:
1. Where is the search bar/input field?
2. What selector should I use to access it?
3. Should I click on it first or can I directly type?

Provide the next action to search for '{product}'.""",

            "search": f"""I've entered the search term '{product}' and this is the search results page.

Please analyze and tell me:
1. Are there product results visible?
2. Which product should I select? (prefer first relevant result)
3. What is the selector for clicking on the product?

Provide the next action to select a product.""",

            "product": f"""This is a product detail page. I need to:
1. Select size '{size}'
2. Add to bag/cart

Please analyze and tell me:
1. Where is the size selector?
2. Is size '{size}' available?
3. Where is the 'Add to Bag' or 'Add to Cart' button?

Provide the next action.""",

            "cart": """I've added the product to the cart. This is the cart/bag page.

Please analyze and tell me:
1. Is the product visible in the cart?
2. Where is the checkout button?
3. Should I proceed to checkout?

Provide the next action to proceed to checkout.""",

            "checkout": """This is the checkout page.

Please analyze and tell me:
1. What information is required?
2. Are we at the payment/review stage?
3. What is the next step?

Note: For this demo, we'll stop before entering actual payment information."""
        }

        base_prompt = prompts.get(current_step, f"Analyze this page for step: {current_step}")

        if page_html:
            base_prompt += f"\n\nHTML SNIPPET (for reference):\n{page_html[:2000]}"

        return base_prompt

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Claude's response into structured action"""
        try:
            # Try to extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            elif "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_str = response_text[json_start:json_end]
            else:
                # Fallback: create structured response from text
                return {
                    "action": "error",
                    "selector": "",
                    "value": "",
                    "reasoning": response_text,
                    "confidence": "low",
                    "alternatives": []
                }

            action_data = json.loads(json_str)

            # Ensure required fields
            required_fields = ["action", "selector", "value", "reasoning"]
            for field in required_fields:
                if field not in action_data:
                    action_data[field] = ""

            if "confidence" not in action_data:
                action_data["confidence"] = "medium"

            if "alternatives" not in action_data:
                action_data["alternatives"] = []

            return action_data

        except json.JSONDecodeError as e:
            return {
                "action": "error",
                "selector": "",
                "value": "",
                "reasoning": f"Failed to parse response: {str(e)}",
                "confidence": "low",
                "alternatives": []
            }

    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
