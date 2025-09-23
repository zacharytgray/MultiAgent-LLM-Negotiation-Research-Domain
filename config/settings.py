"""
Configuration settings for the Multi-Agent Negotiation System.
Centralizes all configurable parameters and constants.
"""

# =============================================================================
# MODEL AND AGENT SETTINGS
# =============================================================================

# Default model settings
DEFAULT_MODEL_NAME = "gpt-oss:20b"
MODEL_TEMPERATURE = 0.5
RESPONSE_TIMEOUT = 60
OLLAMA_BASE_URL = "http://localhost:11434"

# Agent configuration
SYSTEM_INSTRUCTIONS_FILE = "config/system_instructions.txt"
USE_TOOLS_BY_DEFAULT = False

# =============================================================================
# NEGOTIATION SETTINGS
# =============================================================================

# Round settings
DEFAULT_NUM_ROUNDS = 3
DEFAULT_ITEMS_PER_ROUND = 4
MAX_TURNS_PER_ROUND = 30
MAX_RETRIES_PER_INVALID_PROPOSAL = 3
DEFAULT_STARTING_AGENT = 1  # 1 or 2

# Item value constraints
MIN_ITEM_VALUE = 0.0
MAX_ITEM_VALUE = 1.0
ITEM_VALUE_PRECISION = 1  # Number of decimal places

# Available item names (expandable)
ITEM_NAMES = [
    "ItemA", "ItemB", "ItemC", "ItemD", 
    "ItemE", "ItemF", "ItemG", "ItemH",
    "ItemI", "ItemJ", "ItemK", "ItemL"
]

# Agreement detection
AGREEMENT_KEYWORD = "AGREE"
PROPOSAL_KEYWORD = "PROPOSAL"

# =============================================================================
# LOGGING AND OUTPUT SETTINGS
# =============================================================================

# CSV logging
DEFAULT_LOG_DIR = "logs"
DEFAULT_RESULTS_DIR = "results"
CSV_ENCODING = "utf-8"
CSV_DATE_FORMAT = "%Y%m%d"
CSV_TIMESTAMP_FORMAT = "%Y-%m-%d"
CSV_FILENAME_TIMESTAMP_FORMAT = "%Y%m%d_%H%M"  # YYYYMMDD_HHMM for unique filenames

# Console output colors
COLORS = {
    "agent1": "GREEN",
    "agent2": "BLUE", 
    "system": "CYAN",
    "warning": "YELLOW",
    "error": "RED",
    "success": "GREEN",
    "info": "MAGENTA"
}

# Display settings
SEPARATOR_LENGTH = 60
HEADER_SEPARATOR = "="
SECTION_SEPARATOR = "-"

# =============================================================================
# ANALYSIS SETTINGS
# =============================================================================

# Pareto analysis
MAX_IMPROVEMENT_DISPLAY = 3  # Number of improvements to show
EFFICIENCY_THRESHOLDS = {
    "high": 0.9,
    "medium": 0.7,
    "low": 0.0
}

# =============================================================================
# AGENT-SPECIFIC SETTINGS
# =============================================================================

# Boulware agent hyperparameters
BOULWARE_INITIAL_THRESHOLD = 0.85  # Starting threshold percentage (0.0 to 1.0)
BOULWARE_DECREASE_RATE = 0.03      # Amount to decrease threshold per turn
BOULWARE_MIN_THRESHOLD = 0.1       # Minimum threshold (won't go below this)

# =============================================================================
# VALIDATION SETTINGS
# =============================================================================

# Proposal validation
REQUIRE_ALL_ITEMS_ALLOCATED = True
ALLOW_EMPTY_AGENT_ALLOCATION = True
VALIDATE_ITEM_NAMES = True

# =============================================================================
# DEVELOPMENT SETTINGS
# =============================================================================

# Debug and development
DEBUG_MODE = False
VERBOSE_LOGGING = False
SHOW_CONVERSATION_HISTORY = False

# Performance settings
ASYNC_TIMEOUT = 300  # seconds
MAX_MEMORY_MESSAGES = 100  # Limit conversation history length

# =============================================================================
# EXPERIMENTAL FEATURES
# =============================================================================

# Feature flags
ENABLE_TOOL_USE = False
ENABLE_ADVANCED_ANALYSIS = True
ENABLE_REAL_TIME_LOGGING = True

# Experimental thresholds
EXPERIMENTAL_EFFICIENCY_THRESHOLD = 0.95
EXPERIMENTAL_TURN_LIMIT = 15