from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Very simple calculator.

    Format: <number> <op> <number>
    Supported ops: + - * /
    Examples: "2 + 2", "7 * 3", "10 / 4".
    Returns a short result string or an error message.
    """
    parts = expression.strip().split()
    if len(parts) != 3:
        return "Error: use format 'a <op> b' (e.g. 2 + 2)"
    a_str, op, b_str = parts
    try:
        a = float(a_str)
        b = float(b_str)
    except ValueError:
        return "Error: operands must be numbers"

    if op == '+':
        result = a + b
    elif op == '-':
        result = a - b
    elif op == '*':
        result = a * b
    elif op == '/':
        if b == 0:
            return "Error: division by zero"
        result = a / b
    else:
        return "Error: unsupported operator (use + - * /)"

    # Return int form if whole number
    if result.is_integer():
        return f"Result: {int(result)}"
    return f"Result: {result}"


# Function to dynamically collect all tools
def get_all_tools():
    """
    Dynamically collect all functions decorated with @tool in this module.
    Returns a list of all tool functions.
    """
    # Use globals() to get all objects in the current module
    return [calculator]

# Initialize as None, will be populated when first accessed
_ALL_TOOLS = None

def get_tools():
    """Get all tools, populating the cache if needed."""
    global _ALL_TOOLS
    if _ALL_TOOLS is None:
        _ALL_TOOLS = get_all_tools()
    return _ALL_TOOLS

# For backward compatibility - this will be populated after all tools are defined
ALL_TOOLS = None

# This will be called at the end of the module to populate ALL_TOOLS
def _populate_tools():
    global ALL_TOOLS
    ALL_TOOLS = get_tools()

# Call this at the end of the module
_populate_tools()