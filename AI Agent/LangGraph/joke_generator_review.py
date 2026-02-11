from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict

class AgentState(TypedDict):
  topic : str
  joke : str
  review : str

llm = ChatGroq(model="llama-3.1-8b-instant", api_key="YOUR_API_KEY")

def generate_joke(state: AgentState):
  print("Generating Joke...")
  prompt = f"Generate a funny joke about {state['topic']}"
  response = llm.invoke(prompt)
  return {'joke':response.content}

def review_joke(state: AgentState):
  #print("Reviewing Joke....")
  prompt = f"Rate the joke out of 10. Be brutaly honest. The joke {state['joke']}"
  response = llm.invoke(prompt)
  return {'review':response.content}

def main():
  builder = StateGraph(AgentState)

  builder.add_node("generator", generate_joke)
  builder.add_node("reviewer", review_joke)

  builder.add_edge(START, "generator")
  builder.add_edge("generator", "reviewer")
  builder.add_edge("reviewer", END)

  graph = builder.compile()

  initial_state = {'topic':'Gen AI'}
  result = graph.invoke(initial_state)

  print(f"Joke : {result['joke']}")
  print(f"Review : {result['review']}")

if __name__ == '__main__':
  main()
