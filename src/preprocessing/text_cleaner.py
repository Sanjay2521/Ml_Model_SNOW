"""
Text Cleaning Module
Implements all preprocessing steps from the pipeline
"""

import re
import string
import pandas as pd
import numpy as np
from typing import List, Union
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextCleaner:
    """
    Comprehensive text cleaning class for ServiceNow incident data
    """

    def __init__(self, config: dict = None):
        """
        Initialize TextCleaner with configuration

        Args:
            config: Dictionary containing preprocessing configuration
        """
        self.config = config or {}
        self.stop_words = set(stopwords.words('english'))

        # Add custom ServiceNow-specific stop words
        self.custom_stop_words = {
            'please', 'thanks', 'thank', 'regards', 'hi', 'hello',
            'issue', 'problem', 'help', 'need', 'want', 'unable'
        }
        self.stop_words.update(self.custom_stop_words)

    def convert_to_lowercase(self, text: str) -> str:
        """Convert text to lowercase"""
        if pd.isna(text):
            return ""
        return str(text).lower()

    def remove_special_characters(self, text: str) -> str:
        """
        Remove special characters and symbols
        Keep only alphanumeric characters and spaces
        """
        if pd.isna(text):
            return ""
        # Keep letters, numbers, and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        return text

    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        if pd.isna(text):
            return ""
        # Remove http/https URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Remove www URLs
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        return text

    def remove_email_ids(self, text: str) -> str:
        """Remove email addresses from text"""
        if pd.isna(text):
            return ""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        text = re.sub(email_pattern, '', text)
        return text

    def remove_phone_numbers(self, text: str) -> str:
        """Remove phone numbers from text"""
        if pd.isna(text):
            return ""
        # Remove various phone number formats
        phone_patterns = [
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # 123-456-7890
            r'\b\(\d{3}\)\s?\d{3}[-.\s]?\d{4}\b',  # (123) 456-7890
            r'\b\d{10}\b',  # 1234567890
            r'\+\d{1,3}[-.\s]?\d{1,14}\b'  # International format
        ]
        for pattern in phone_patterns:
            text = re.sub(pattern, '', text)
        return text

    def remove_file_paths(self, text: str) -> str:
        """Remove file paths (Windows and Unix)"""
        if pd.isna(text):
            return ""
        # Windows paths
        text = re.sub(r'[a-zA-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*', '', text)
        # Unix paths
        text = re.sub(r'/(?:[^/\s]+/)*[^/\s]*', '', text)
        return text

    def remove_extra_spaces(self, text: str) -> str:
        """Remove extra spaces, tabs, and newlines"""
        if pd.isna(text):
            return ""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading and trailing spaces
        text = text.strip()
        return text

    def remove_stop_words(self, text: str) -> str:
        """Remove stop words from text"""
        if pd.isna(text):
            return ""
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
        return ' '.join(filtered_tokens)

    def remove_common_words(self, text: str, common_words: set = None) -> str:
        """
        Remove common words specific to the context

        Args:
            text: Input text
            common_words: Set of common words to remove
        """
        if pd.isna(text):
            return ""
        if common_words is None:
            # Default ServiceNow common words
            common_words = {
                'incident', 'ticket', 'request', 'service', 'snow',
                'servicenow', 'system', 'application', 'user'
            }

        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in common_words]
        return ' '.join(filtered_tokens)

    def remove_numbers(self, text: str) -> str:
        """Remove standalone numbers"""
        if pd.isna(text):
            return ""
        text = re.sub(r'\b\d+\b', '', text)
        return text

    def remove_single_characters(self, text: str) -> str:
        """Remove single character words"""
        if pd.isna(text):
            return ""
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if len(word) > 1]
        return ' '.join(filtered_tokens)

    def clean_text(self, text: str, full_clean: bool = True) -> str:
        """
        Apply all cleaning steps to text

        Args:
            text: Input text
            full_clean: If True, apply all cleaning steps

        Returns:
            Cleaned text
        """
        if pd.isna(text) or text == "":
            return ""

        # Step 1: Convert to lowercase
        if self.config.get('lowercase', True):
            text = self.convert_to_lowercase(text)

        # Step 2: Remove URLs
        if self.config.get('remove_urls', True):
            text = self.remove_urls(text)

        # Step 3: Remove email IDs
        if self.config.get('remove_emails', True):
            text = self.remove_email_ids(text)

        # Step 4: Remove phone numbers
        if self.config.get('remove_phone_numbers', True):
            text = self.remove_phone_numbers(text)

        # Step 5: Remove file paths
        if self.config.get('remove_file_paths', True):
            text = self.remove_file_paths(text)

        # Step 6: Remove special characters
        if self.config.get('remove_special_chars', True):
            text = self.remove_special_characters(text)

        # Step 7: Remove extra spaces
        if self.config.get('remove_extra_spaces', True):
            text = self.remove_extra_spaces(text)

        if full_clean:
            # Step 8: Remove stop words
            if self.config.get('remove_stop_words', True):
                text = self.remove_stop_words(text)

            # Step 9: Remove numbers (optional)
            if self.config.get('remove_numbers', False):
                text = self.remove_numbers(text)

            # Step 10: Remove single characters
            if self.config.get('remove_single_chars', True):
                text = self.remove_single_characters(text)

            # Step 11: Final space cleanup
            text = self.remove_extra_spaces(text)

        return text

    def clean_dataframe(self, df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame:
        """
        Clean text columns in a dataframe

        Args:
            df: Input dataframe
            text_columns: List of text column names to clean

        Returns:
            Dataframe with cleaned text columns
        """
        df_clean = df.copy()

        for col in text_columns:
            if col in df_clean.columns:
                print(f"Cleaning column: {col}")
                df_clean[col] = df_clean[col].apply(self.clean_text)

                # Create combined cleaned text column if multiple text columns
                if len(text_columns) > 1:
                    if 'cleaned_text' not in df_clean.columns:
                        df_clean['cleaned_text'] = df_clean[col]
                    else:
                        df_clean['cleaned_text'] = df_clean['cleaned_text'] + ' ' + df_clean[col]

        # Final cleanup of combined text
        if 'cleaned_text' in df_clean.columns:
            df_clean['cleaned_text'] = df_clean['cleaned_text'].apply(self.remove_extra_spaces)

        return df_clean


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from dataframe"""
    print(f"Shape before removing duplicates: {df.shape}")
    df_clean = df.drop_duplicates()
    print(f"Shape after removing duplicates: {df_clean.shape}")
    print(f"Removed {df.shape[0] - df_clean.shape[0]} duplicate rows")
    return df_clean


def handle_null_values(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handle null values in dataframe

    Args:
        df: Input dataframe
        strategy: 'drop', 'fill', or 'fillna'

    Returns:
        Dataframe with handled null values
    """
    print(f"Null values before handling:\n{df.isnull().sum()}")

    if strategy == 'drop':
        # Drop rows with null values in important columns
        df_clean = df.dropna(subset=['short_description', 'description'], how='all')
    elif strategy == 'fill':
        # Fill null values with empty string
        df_clean = df.fillna('')
    else:
        df_clean = df.copy()

    print(f"\nNull values after handling:\n{df_clean.isnull().sum()}")
    return df_clean


def drop_irrelevant_columns(df: pd.DataFrame, columns_to_keep: List[str] = None) -> pd.DataFrame:
    """
    Drop columns not relevant for incident assignment

    Args:
        df: Input dataframe
        columns_to_keep: List of columns to keep

    Returns:
        Dataframe with only relevant columns
    """
    if columns_to_keep is None:
        # Default columns relevant for incident assignment
        columns_to_keep = [
            'number', 'short_description', 'description', 'close_notes',
            'assignment_group', 'assigned_to', 'priority', 'impact',
            'urgency', 'category', 'subcategory', 'state', 'opened_at',
            'closed_at', 'resolved_at'
        ]

    # Keep only columns that exist in the dataframe
    existing_columns = [col for col in columns_to_keep if col in df.columns]

    print(f"Keeping {len(existing_columns)} columns out of {len(df.columns)}")
    return df[existing_columns]


if __name__ == "__main__":
    # Example usage
    cleaner = TextCleaner()

    sample_text = """
    Hi Team, I am unable to access my email application.
    Error: Connection timeout at https://mail.company.com
    My email is john.doe@company.com and phone is 123-456-7890.
    File path: C:\\Users\\john\\Documents\\error.log
    Please help!!! Thanks.
    """

    cleaned = cleaner.clean_text(sample_text)
    print("Original text:")
    print(sample_text)
    print("\nCleaned text:")
    print(cleaned)
