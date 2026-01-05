import os
import glob
import google.generativeai as genai
from getpass import getpass
from google.colab import files
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass("Enter your Google API Key: ")
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("\n🔍 Checking your available models...")
valid_model = None
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            # We prefer 1.5 Flash, but will accept Pro or others
            if 'gemini-1.5-flash' in m.name:
                valid_model = m.name
                break
            elif 'gemini-pro' in m.name and valid_model is None:
                valid_model = m.name
    
    if valid_model:
        valid_model = valid_model.replace('models/', '')
        print(f"✅ Found working model: {valid_model}")
    else:
        valid_model = "gemini-pro" # Ultimate fallback
        print("⚠️ Could not auto-detect, defaulting to 'gemini-pro'")
        
except Exception as e:
    print(f"Warning: Model check failed ({e}). Defaulting to 'gemini-1.5-flash'.")
    valid_model = "gemini-1.5-flash"

list_of_files = glob.glob('*.pdf')
if list_of_files:
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"📂 Processing file: {latest_file}")
else:
    print("\n👉 Please upload your PDF now:")
    uploaded = files.upload()
    if not uploaded:
        raise SystemExit("No file uploaded.")
    latest_file = next(iter(uploaded))

loader = PyPDFLoader(latest_file)
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(pages)

print("🧠 Building Vector Database...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(docs, embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

llm = ChatGoogleGenerativeAI(
    model=valid_model, 
    temperature=0.3
)

template = """Answer the question based ONLY on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), 
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


print("\n" + "="*30)
print(f" 🤖 RAG READY (Model: {valid_model}) ")
print("="*30)

while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == 'exit':
        break
    try:
        print("Answer: ", end="")
        for chunk in rag_chain.stream(query):
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
