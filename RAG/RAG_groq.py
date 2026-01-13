from langchain_community.document_loaders import PyPDFLoader
from google.colab import userdata
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings


token = userdata.get('Groq_api_key')

document_pdf="/content/The_Stranger_-_Albert_Camus.pdf"
print("Loading Docs.....")
loader = PyPDFLoader(document_pdf)
docs=loader.load()

print("Text splitting...")
text_split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_split.split_documents(docs)

print("embedding")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLm-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
retriever = vectorstore.as_retriever()

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    temperature = 0.7
)

template= """
You are a helpful assistant. Answer the user's question based ONLY on the context below. 
If the answer is not in the context, say "I don't know based on this document." and fetch answer from LLM

Context:
{context}

Question: {question}
"""
prompt =ChatPromptTemplate.from_template(template)

while True:
      user_query = input("\nAsk a question about your PDF: ")
    
      if user_query.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break

      retrieved_docs = retriever.invoke(user_input)
      context_data = "\n\n".join([doc.page_content for doc in retriever_docs ])

      chain = prompt | llm
      response = chain.invoke({"context":context_data, "question":user_query})

      print(f"AI: \n {response.content}")
