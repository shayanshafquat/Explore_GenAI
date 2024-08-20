from dotenv import load_dotenv
import os
from os.path import dirname
import sqlite3
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion


# Load environment variables from .env file
load_dotenv()
deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_key = os.environ["AZURE_OPENAI_API_KEY"]

# Database setup
db_path = os.path.join('conversation_history.db')
conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS conversation_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT NOT NULL,
        content TEXT NOT NULL
    )
''')
conn.commit()

# Function to save conversation to the database
def save_to_db(role, content):
    c.execute('INSERT INTO conversation_history (role, content) VALUES (?, ?)', (role, content))
    conn.commit()

async def main():
    kernel = sk.Kernel()

    kernel.add_chat_service("dv", AzureChatCompletion(
        deployment_name=deployment_name,
        endpoint=endpoint,
        api_key=api_key)
    )

    # Define multiple conversation flows
    # Flow 1: Espresso Variables
    espresso_prompt = kernel.create_semantic_function("""
    I need to understand what are the variables involved in making outstanding espresso besides
    a good machine. For example what is the combination of roast, grind, tamp, and water temperature.
    Include 3 practical steps to practice and improve each variable.
    """)

    # Flow 2: Coffee Beans
    coffee_beans_prompt = kernel.create_semantic_function("""
    Explain the differences between Arabica and Robusta coffee beans, including their flavor profiles,
    growing conditions, and common uses.
    """)

    # Flow 3: Brewing Methods
    brewing_methods_prompt = kernel.create_semantic_function("""
    Describe different methods of brewing coffee, such as French press, pour-over, and espresso. 
    Explain the unique characteristics of each method and provide tips for achieving the best results.
    """)

    # Function to run a specific conversation flow
    def run_conversation_flow(prompt_function):
        response = prompt_function()
        return response.result

    # User selection of conversation flow
    print("Choose a conversation flow:")
    print("1. Espresso Variables")
    print("2. Coffee Beans")
    print("3. Brewing Methods")
    choice = input("Enter the number of your choice: ")

    if choice == '1':
        save_to_db('user', 'Selected: Espresso Variables')
        response = run_conversation_flow(espresso_prompt)
        print(response)
        save_to_db('assistant', response)
    elif choice == '2':
        save_to_db('user', 'Selected: Coffee Beans')
        response = run_conversation_flow(coffee_beans_prompt)
        print(response)
        save_to_db('assistant', response)
    elif choice == '3':
        save_to_db('user', 'Selected: Brewing Methods')
        response = run_conversation_flow(brewing_methods_prompt)
        print(response)
        save_to_db('assistant', response)
    else:
        print("Invalid choice. Exiting.")
        save_to_db('user', 'Selected: Invalid choice')

# Run the main function
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
    conn.close()