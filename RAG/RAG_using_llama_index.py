from google.colab import userdata
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings

api = userdata.get('GROQ_API_KEY')

# Setting up the LLM
Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=api)

# Embedding Model im using Huggingface
Settings.embed_model = HuggingFaceEmbedding(
    model_name = "BAAI/bge-small-en-v1.5"
)

# Loading Documents
print("Loading Document...")
document = SimpleDirectoryReader(
    input_files=["/content/The_Stranger_-_Albert_Camus.pdf"]
).load_data()

# Vector index initializing
print("Initializing text chunking and embedding")
chunks = VectorStoreIndex.from_documents(document)

# Setting up Query Engine
quey_engine = chunks.as_query_engine(streaming=True)

# User query Interaction
print("Enter quit or exit to stop the program")
while True:
  user_input = input("Ask the question  :")
  if user_input.lower() in ['quit','exit']:
    print("Goodbye!!")
    break

  # Getting response from model
  response = query_engine.query(user_input)
  print("Model thinking...")
  print("AI : ")
  response.print_response_stream()



