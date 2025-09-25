from langchain.prompts import ChatPromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from colorama import Fore
from src.utils.ThinkingModelProcessor import strip_thinking_blocks, has_thinking_blocks

class Agent:
    def __init__(self, modelName, instructionsFilepath, use_tools: bool = False):
        self.modelName = modelName
        self.instructionsFilpath = instructionsFilepath
        self.memory = [self.importInstructions()]
        self.temperature = 0.5
        self.responseTimeout = 60
        self.model = ChatOllama(model=self.modelName, base_url="http://localhost:11434", temperature=self.temperature)

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
        else:
            raise ValueError(f"Unknown role: {role}")
    
    def reset_memory(self):
        """Reset memory to only contain the original system instructions."""
        original_system_message = self.importInstructions()
        self.memory = [original_system_message] 

    async def generateResponse(self, inputTextRole=None, inputText=None): # Generate response based on input
        try:
            if inputText and inputTextRole:
                self.addToMemory(inputTextRole, inputText)

            history = ChatPromptTemplate.from_messages(self.memory)
            chain = history | self.model
            response = await chain.ainvoke({})

            self.addToMemory('assistant', response)

            # Get the raw response content
            raw_content = response.content.strip()
            
            # Automatically strip thinking blocks if present
            if has_thinking_blocks(raw_content):
                cleaned_content = strip_thinking_blocks(raw_content)
                return cleaned_content
            else:
                return raw_content
        except Exception as e:
            print(f"{Fore.RED}Error generating response: {e}{Fore.RESET}")
            return None

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
            else:
                print(f"Unknown message type: {message}")
            print("----------------------------------------------------------------------------------------")
        print(f"----------------{Fore.LIGHTYELLOW_EX}END History:{Fore.RESET}----------------")