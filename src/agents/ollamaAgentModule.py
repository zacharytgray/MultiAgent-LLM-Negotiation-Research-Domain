from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from colorama import Fore
import re
from .ollamaTools import ALL_TOOLS  # Import dynamic tool collection
from langchain_core.messages import ToolMessage
from src.utils.ThinkingModelProcessor import (
    strip_thinking_blocks, 
    has_thinking_blocks, 
    has_malformed_thinking_blocks, 
    is_response_too_short_after_thinking_removal
)

class Agent:
    def __init__(self, modelName, instructionsFilepath, use_tools: bool = False):
        self.modelName = modelName
        self.instructionsFilpath = instructionsFilepath
        self.memory = [self.importInstructions()]
        self.temperature = 0.5
        self.responseTimeout = 60
        self.use_tools = use_tools
        self.tools = ALL_TOOLS if use_tools else []
        self.model = ChatOllama(model=self.modelName, base_url="http://localhost:11434", temperature=self.temperature)
        if self.use_tools and self.tools:
            self.model = self.model.bind_tools(self.tools)            

    def importInstructions(self):
        instructions = ""
        try:
            with open(self.instructionsFilpath, 'r') as f:
                instructions += f.read()
        except FileNotFoundError:
            print(f"{Fore.RED}Instructions file not found: {self.instructionsFilpath}{Fore.RESET}")
            exit(1)
        return SystemMessage(content=instructions)

    def addToMemory(self, role, content, tool_call_id=None):
        if role == 'system':
            self.memory.append(SystemMessage(content=content))
        elif role == 'user':
            self.memory.append(HumanMessage(content=content))
        elif role == 'assistant':
            # If content is an AIMessage, add it directly. Otherwise, create a new one.
            if isinstance(content, AIMessage):
                self.memory.append(content)
            else:
                self.memory.append(AIMessage(content=content))
        elif role == 'tool':
            if tool_call_id is None:
                raise ValueError("tool_call_id must be provided for role 'tool'")
            self.memory.append(ToolMessage(content=content, tool_call_id=tool_call_id))
        else:
            raise ValueError(f"Unknown role: {role}")
    
    def reset_memory(self):
        """Reset memory to only contain the original system instructions."""
        original_system_message = self.importInstructions()
        self.memory = [original_system_message] 

    async def generateResponse(self, inputTextRole=None, inputText=None, max_thinking_retries=3): # Generate response based on input
        try:
            if inputText and inputTextRole:
                self.addToMemory(inputTextRole, inputText)

            # Retry loop for malformed thinking blocks
            for attempt in range(max_thinking_retries + 1):
                # Create memory snapshot before this attempt (don't add failed attempts to permanent memory)
                memory_snapshot = self.memory.copy()
                
                history = ChatPromptTemplate.from_messages(self.memory)
                chain = history | self.model
                response = await chain.ainvoke({})
                
                # Handle tool calls if present and tools are enabled
                if self.use_tools and hasattr(response, 'tool_calls') and response.tool_calls:
                    print(f"{Fore.YELLOW}Tool calls detected: {len(response.tool_calls)}\nResponse: {response.content}{Fore.RESET}")
                    
                    # Add the assistant's response (which contains tool calls) to memory
                    self.addToMemory('assistant', response)
                    
                    # Execute each tool call
                    for tool_call in response.tool_calls:
                        print(f"{Fore.CYAN}Executing tool: {tool_call['name']} with args: {tool_call['args']}{Fore.RESET}")
                        tool_output = self.execute_tool(tool_call)
                        self.addToMemory('tool', tool_output, tool_call['id'])
                    
                    # Generate a new response with the tool results
                    history = ChatPromptTemplate.from_messages(self.memory)
                    chain = history | self.model
                    response = await chain.ainvoke({})

                # Get the raw response content
                raw_content = response.content.strip()
                
                # Validate thinking blocks
                if has_malformed_thinking_blocks(raw_content):
                    print(f"{Fore.YELLOW}Attempt {attempt + 1}: Detected malformed thinking blocks{Fore.RESET}")
                    print(f"{Fore.YELLOW}Raw response preview: {raw_content[:200]}...{Fore.RESET}")
                    
                    # If this isn't the last attempt, restore memory and retry
                    if attempt < max_thinking_retries:
                        print(f"{Fore.CYAN}Retrying LLM call (attempt {attempt + 2}/{max_thinking_retries + 1})...{Fore.RESET}")
                        self.memory = memory_snapshot  # Restore memory state
                        continue
                    else:
                        print(f"{Fore.RED}Max retries reached for malformed thinking blocks, attempting recovery{Fore.RESET}")
                        # Add the response to memory and try to recover
                        self.addToMemory('assistant', response)
                        return self._handle_malformed_thinking_blocks(raw_content)
                
                # Check if response becomes too short after thinking removal
                if is_response_too_short_after_thinking_removal(raw_content, min_length=10):
                    print(f"{Fore.YELLOW}Attempt {attempt + 1}: Response too short after thinking block removal{Fore.RESET}")
                    print(f"{Fore.YELLOW}Raw response: {raw_content}{Fore.RESET}")
                    
                    # If this isn't the last attempt, restore memory and retry
                    if attempt < max_thinking_retries:
                        print(f"{Fore.CYAN}Retrying LLM call for short response (attempt {attempt + 2}/{max_thinking_retries + 1})...{Fore.RESET}")
                        self.memory = memory_snapshot  # Restore memory state
                        continue
                    else:
                        print(f"{Fore.RED}Max retries reached for short responses, using fallback{Fore.RESET}")
                        # Add the response to memory and return fallback
                        self.addToMemory('assistant', response)
                        return "I need to think more about this proposal. Could you provide more details?"
                
                # Success! Add to memory and process normally
                self.addToMemory('assistant', response)
                
                # Normal processing
                if has_thinking_blocks(raw_content):
                    cleaned_content = strip_thinking_blocks(raw_content)
                    print(f"{Fore.GREEN}Successfully processed thinking blocks (attempt {attempt + 1}){Fore.RESET}")
                    return cleaned_content
                else:
                    print(f"{Fore.GREEN}Clean response received (attempt {attempt + 1}){Fore.RESET}")
                    return raw_content
            
        except Exception as e:
            print(f"{Fore.RED}Error generating response: {e}{Fore.RESET}")
            return None
    
    
    def _handle_malformed_thinking_blocks(self, raw_content: str) -> str:
        """
        Handle responses with malformed thinking blocks (unclosed <think> tags).
        
        Args:
            raw_content: Raw response with malformed thinking blocks
            
        Returns:
            str: Best attempt at extracting actual response content
        """
        # Find the last <think> tag and treat everything after it as thinking content
        # Look for patterns that might indicate the start of actual response content
        
        # Strategy 1: Look for common response patterns after <think>
        patterns = [
            r'<think>.*?(?=I propose|I accept|I counter|I suggest|I offer|PROPOSAL|AGREE|REJECT)',
            r'<think>.*?(?=\{[^}]*"agent[12]")',  # JSON proposals
            r'<think>.*?(?=Let me|I think|I believe|My proposal)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, raw_content, flags=re.DOTALL | re.IGNORECASE)
            if match:
                # Extract everything after the thinking content
                actual_response = raw_content[match.end():]
                if len(actual_response.strip()) > 10:  # Must have substantial content
                    print(f"{Fore.CYAN}Recovered response from malformed thinking blocks{Fore.RESET}")
                    return actual_response.strip()
        
        # Strategy 2: If no clear pattern, look for the last meaningful sentence
        # Split by sentences and take the last few that might be actual response
        sentences = re.split(r'[.!?]+', raw_content)
        if len(sentences) > 1:
            # Take the last 2-3 sentences as potential actual response
            potential_response = '. '.join(sentences[-3:]).strip()
            if len(potential_response) > 20:
                print(f"{Fore.CYAN}Extracted potential response from end of malformed thinking block{Fore.RESET}")
                return potential_response
        
        # Fallback: Return a safe default response
        print(f"{Fore.RED}Unable to recover meaningful content from malformed thinking blocks{Fore.RESET}")
        return "I need to reconsider this proposal. Could you clarify your offer?"
    
    def execute_tool(self, tool_call):
        """Execute a tool call and return the result"""
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    result = tool.invoke(tool_args)
                    print(f"{Fore.GREEN}Tool {tool_name} executed successfully: {result}{Fore.RESET}")
                    return str(result)
                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {str(e)}"
                    print(f"{Fore.RED}{error_msg}{Fore.RESET}")
                    return error_msg
        
        error_msg = f"Tool {tool_name} not found."
        print(f"{Fore.RED}{error_msg}{Fore.RESET}")
        return error_msg
            
    def printMemory(self, skipSystemMessage=False):
        if skipSystemMessage:
            messages_to_print = [msg for msg in self.memory if not isinstance(msg, SystemMessage)]
        else:
            messages_to_print = self.memory
        print(f"----------------{Fore.LIGHTYELLOW_EX}Conversation History:{Fore.RESET}----------------")
        for message in messages_to_print:
            if isinstance(message, SystemMessage):
                print(f"{Fore.LIGHTRED_EX}System: {message.content}{Fore.RESET}")
            elif isinstance(message, HumanMessage):
                print(f"{Fore.LIGHTGREEN_EX}User: {message.content}{Fore.RESET}")
            elif isinstance(message, AIMessage):
                print(f"{Fore.LIGHTBLUE_EX}Agent: {message.content}{Fore.RESET}")
                # Show tool calls if present
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        print(f"{Fore.LIGHTBLUE_EX}Tool Call: {tool_call['name']}({tool_call['args']}){Fore.RESET}")
            elif isinstance(message, ToolMessage):
                print(f"{Fore.YELLOW}Tool Output: {message.content}{Fore.RESET}")
            else:
                print(f"Unknown message type: {message}")
            print("----------------------------------------------------------------------------------------")
        print(f"----------------{Fore.LIGHTYELLOW_EX}END History:{Fore.RESET}----------------")