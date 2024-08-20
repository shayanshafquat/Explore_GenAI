from dotenv import load_dotenv
import os
from os.path import dirname
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
import sqlite3

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

    # Function to generate a response based on the current conversation history
    async def generate_response(prompt):
        messages = ""
        for row in c.execute('SELECT role, content FROM conversation_history ORDER BY id'):
            role = "User" if row[0] == "user" else "Assistant"
            messages += f"{role}: {row[1]}\n"
        messages += f"User: {prompt}\nAssistant: "
        
        semantic_function = kernel.create_semantic_function(messages)
        response = semantic_function()
        return response.result

    # User interaction loop
    while True:
        user_message = input("You: ")
        if user_message.lower() == 'exit':
            break
        
        save_to_db('user', user_message)
        
        assistant_response = await generate_response(user_message)
        print(f"Assistant: {assistant_response}")
        save_to_db('assistant', assistant_response)

# Run the main function
if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
    conn.close()