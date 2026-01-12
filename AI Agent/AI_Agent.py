import requests
import json
import time
import datetime
from math import * 
from google.colab import userdata


token = userdata.get('HF_TOKEN')
HF_TOKEN = token

API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}
MODEL_ID = "google/gemma-2-2b-it"

def check_time():
  now = datetime.datetime.now()
  return now.strftime("%Y-%m-%d %H:%M:%S")

def calc_now(exp):
  #just = str(exp)
  return str(eval(exp))

def query(messages):
  payload = {
      "model" : MODEL_ID,
      "messages" : messages,
      "max_tokens" : 500,
      "temperature":0.4
  }

  response = requests.post(API_URL, headers= HEADERS, json = payload)

  if response.status_code == 200:
    return response.json()['choices'][0]['message']['content']
  else:
    print("error in model")


system_instruction = """
You are a functional AI Agent.
RULES:
1. If the user asks for Time/Date, output strictly: TOOL: TIME
2. If the user asks for Math, output strictly: TOOL: CALC [expression]
3. If you receive a 'Tool Result', you MUST use it to formulate a final natural language answer to the user. Do not say 'Okay' or 'I understand'. Just give the answer.
"""

messages = [
    {"role" : "system", "content" : system_instruction}
]

print("Type exit or quit to exit....")
while True:
  user_input = input('YOU:')
  if user_input.lower() in ['quit', 'exit']:
    break

  messages.append({"role":"user","content":user_input})

  ai_response = query(messages)

  if 'TOOL: TIME' in ai_response:
    tool_result = check_time()

    messages.append({"role":"system", "content":ai_response})
    # 2. Check for Tools
    messages.append({
            "role": "user",
            "content": f"Tool Result: {tool_result}. Use this result to explicitly answer my question: '{user_input}'."
        })

    data= query(messages)
    print("Agent :", data)

  elif "TOOL: CALC" in ai_response:
    tool_result = calc_now(user_input)

    messages.append({"role":"system","content":ai_response})
    messages.append({
            "role": "user",
            "content": f"Tool Result: {tool_result}. Use this result to explicitly answer my question: '{user_input}'."
    })

    data=query(messages)
    print(f"Calc Agent: {data}")

  else:
    print("Convo", ai_response)
    messages.append({"role":"user", "content":ai_response})

