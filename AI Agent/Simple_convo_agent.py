import requests
import json
import time
from google.colab import userdata


token = userdata.get('HF_TOKEN')
HF_TOKEN = token

# 1. SETUP: 
API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# 2. MODEL: 
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct:cerebras"

def query_huggingface_router(messages):
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.7
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)

    # ERROR HANDLING 
    if response.status_code != 200:
        print(f"\n[!] Error {response.status_code}: {response.text}")

        # If model is loading (503), wait and retry
        if response.status_code == 503:
            print("(Model is warm-up loading... waiting 10s)")
            time.sleep(10)
            return query_huggingface_router(messages)

        return None

    return response.json()

print("Type 'quit' to exit.\n")

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."}
]

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("Exiting...")
        break

    messages.append({"role": "user", "content": user_input})

    try:
        print("Bot: ", end="", flush=True)

        data = query_huggingface_router(messages)

        if data and 'choices' in data:
            bot_response = data['choices'][0]['message']['content']
            print(bot_response)
            print("\n")
            messages.append({"role": "assistant", "content": bot_response})
        else:
            print(" (No valid response)")

    except Exception as e:
        print(f"\nError: {e}")
