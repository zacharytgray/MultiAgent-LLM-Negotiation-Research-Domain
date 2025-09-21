from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from colorama import Fore
from ollamaTools import ALL_TOOLS  # Import dynamic tool collection
from langchain_core.messages import ToolMessage

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

    async def generateResponse(self, inputTextRole=None, inputText=None): # Generate response based on input
        try:
            if inputText and inputTextRole:
                self.addToMemory(inputTextRole, inputText)

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

            self.addToMemory('assistant', response)

            return response.content.strip()
        except Exception as e:
            print(f"{Fore.RED}Error generating response: {e}{Fore.RESET}")
            return None
    
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