"""
Simplified Swiss text formatter - Only essential functions
"""
import re
from typing import List, Tuple


def format_swiss_numbers(text: str) -> str:
    """
    Format numbers in text to Swiss standard (1'234'567.89)
    
    Args:
        text: Input text containing numbers
        
    Returns:
        Text with Swiss-formatted numbers
    """
    if not text:
        return text
    
    # Patterns for different number formats
    patterns = [
        # Standard numbers with optional decimals
        (r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', _format_standard_number),
        # Numbers already with apostrophes (validate format)
        (r"\b\d{1,3}(?:'\d{3})*(?:\.\d+)?\b", _validate_swiss_number),
        # Percentages
        (r'\b\d+(?:\.\d+)?%', _format_percentage),
        # Currency amounts
        (r'(?:CHF|EUR|USD)\s*\d+(?:[,.\s]\d{3})*(?:[.,]\d+)?', _format_currency)
    ]
    
    # Apply each pattern
    for pattern, formatter in patterns:
        text = re.sub(pattern, formatter, text)
    
    return text


def _format_standard_number(match: re.Match) -> str:
    """Format a standard number to Swiss format"""
    number = match.group(0)
    
    # Skip years (4 digits starting with 19 or 20)
    if re.match(r'^(19|20)\d{2}$', number):
        return number
    
    # Remove existing separators
    clean = number.replace(',', '').replace("'", '')
    
    # Split into integer and decimal parts
    parts = clean.split('.')
    integer_part = parts[0]
    decimal_part = parts[1] if len(parts) > 1 else None
    
    # Format integer part with apostrophes
    if len(integer_part) > 3:
        # Add apostrophes from right to left
        formatted = ''
        for i, digit in enumerate(reversed(integer_part)):
            if i > 0 and i % 3 == 0:
                formatted = "'" + formatted
            formatted = digit + formatted
        integer_part = formatted
    
    # Reconstruct number
    if decimal_part:
        return f"{integer_part}.{decimal_part}"
    return integer_part


def _validate_swiss_number(match: re.Match) -> str:
    """Validate and fix Swiss formatted numbers"""
    number = match.group(0)
    # If it's already properly formatted, return as is
    if re.match(r"^\d{1,3}(?:'\d{3})*(?:\.\d+)?$", number):
        return number
    # Otherwise reformat
    return _format_standard_number(match)


def _format_percentage(match: re.Match) -> str:
    """Ensure proper spacing for percentages"""
    percentage = match.group(0)
    # Add space before % if not present
    if not percentage[-2].isspace():
        return percentage[:-1] + ' %'
    return percentage


def _format_currency(match: re.Match) -> str:
    """Format currency amounts"""
    amount = match.group(0)
    
    # Extract currency and number
    currency_match = re.match(r'(CHF|EUR|USD)\s*(.+)', amount)
    if not currency_match:
        return amount
    
    currency = currency_match.group(1)
    number_str = currency_match.group(2)
    
    # Clean and format the number
    clean_number = re.sub(r'[^\d.,]', '', number_str)
    clean_number = clean_number.replace(',', '.')
    
    try:
        # Convert to float and back to get proper decimal places
        value = float(clean_number)
        
        # Format with Swiss thousands separator
        if value >= 1000:
            formatted = f"{value:,.2f}".replace(',', "'")
        else:
            formatted = f"{value:.2f}"
        
        return f"{currency} {formatted}"
    except ValueError:
        return amount


# Simple tests
if __name__ == "__main__":
    test_cases = [
        "Der Gewinn betrug 1234567.89 CHF",
        "Umsatzsteigerung von 15.5%",
        "Im Jahr 2023 waren es CHF 1,234,567",
        "Die Marge liegt bei 12.5 %",
        "EUR 999 und USD 1234567"
    ]
    
    for test in test_cases:
        formatted = format_swiss_numbers(test)
        print(f"Original: {test}")
        print(f"Formatted: {formatted}")
        print()