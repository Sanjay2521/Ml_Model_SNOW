"""Unit tests for preprocessing module"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
from src.preprocessing import TextCleaner


class TestTextCleaner:
    """Test TextCleaner class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.cleaner = TextCleaner()

    def test_convert_to_lowercase(self):
        """Test lowercase conversion"""
        text = "Hello WORLD"
        result = self.cleaner.convert_to_lowercase(text)
        assert result == "hello world"

    def test_remove_urls(self):
        """Test URL removal"""
        text = "Check out https://example.com for more info"
        result = self.cleaner.remove_urls(text)
        assert "https://example.com" not in result

    def test_remove_email_ids(self):
        """Test email removal"""
        text = "Contact me at john.doe@example.com"
        result = self.cleaner.remove_email_ids(text)
        assert "john.doe@example.com" not in result

    def test_remove_phone_numbers(self):
        """Test phone number removal"""
        text = "Call me at 123-456-7890"
        result = self.cleaner.remove_phone_numbers(text)
        assert "123-456-7890" not in result

    def test_clean_text(self):
        """Test full text cleaning"""
        text = "Hi! Email: test@example.com, Phone: 123-456-7890, URL: https://test.com"
        result = self.cleaner.clean_text(text)
        assert "test@example.com" not in result
        assert "123-456-7890" not in result
        assert "https://test.com" not in result


if __name__ == "__main__":
    pytest.main([__file__])
