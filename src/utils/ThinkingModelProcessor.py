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