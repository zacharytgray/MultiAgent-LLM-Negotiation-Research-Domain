"""
Utility for processing responses from thinking models that use <think> blocks.
"""

import re


def strip_thinking_blocks(response: str) -> str:
    """
    Remove <think>...</think> blocks from model responses.
    
    Args:
        response: Raw response from the model
        
    Returns:
        str: Response with thinking blocks removed and whitespace cleaned up
    """
    # Remove <think>...</think> blocks (supports multiline)
    cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up any extra whitespace that might be left
    cleaned_response = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_response)  # Multiple newlines -> double newline
    cleaned_response = cleaned_response.strip()  # Remove leading/trailing whitespace
    
    return cleaned_response


def has_thinking_blocks(response: str) -> bool:
    """
    Check if a response contains thinking blocks.
    
    Args:
        response: Response to check
        
    Returns:
        bool: True if response contains <think> blocks
    """
    return bool(re.search(r'<think>.*?</think>', response, flags=re.DOTALL | re.IGNORECASE))


def has_malformed_thinking_blocks(response: str) -> bool:
    """
    Check if a response contains malformed thinking blocks (unclosed <think> tags).
    
    Args:
        response: Response to check
        
    Returns:
        bool: True if response has unclosed <think> tags
    """
    # Check for <think> without matching </think>
    think_opens = len(re.findall(r'<think>', response, flags=re.IGNORECASE))
    think_closes = len(re.findall(r'</think>', response, flags=re.IGNORECASE))
    
    return think_opens > think_closes


def is_response_too_short_after_thinking_removal(response: str, min_length: int = 10) -> bool:
    """
    Check if response becomes too short after removing thinking blocks.
    
    Args:
        response: Response to check
        min_length: Minimum acceptable length for actual content
        
    Returns:
        bool: True if response is too short after thinking block removal
    """
    if has_thinking_blocks(response):
        cleaned = strip_thinking_blocks(response)
        return len(cleaned.strip()) < min_length
    return False