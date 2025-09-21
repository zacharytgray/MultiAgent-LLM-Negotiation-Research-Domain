import asyncio
from colorama import Fore
from ollamaAgentModule import Agent

def checkForChatExit(userInput):
    if userInput.lower() in ['exit', 'quit', 'q']:
        return True
    return False

async def main():
    model_name = "gpt-oss:20b"
    instructions_filepath = "systemInstructions.txt"
    agent = Agent(modelName=model_name, instructionsFilepath=instructions_filepath, use_tools = True)
    chatIsActive = True
    while chatIsActive:
        userInput = input(f"{Fore.LIGHTGREEN_EX}You: {Fore.RESET}")

        if checkForChatExit(userInput): # Check if user wants to exit chat
            chatIsActive = False
            print(f"{Fore.YELLOW}Exiting chat...{Fore.RESET}")
            continue
        elif userInput.lower() == 'history':
            agent.printMemory()
            continue
        elif userInput.lower() == 'clear':
            agent.memory = [agent.memory[0]]
        else:
            await agent.generateResponse(inputTextRole='user', inputText=userInput)
            print(f"{Fore.LIGHTCYAN_EX}Assistant: {Fore.RESET}{agent.memory[-1].content.strip()}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"{Fore.YELLOW}Exiting gracefully...{Fore.RESET}") 
    except Exception as e:
        print(f"{Fore.RED}An error occurred: {e}{Fore.RESET}")