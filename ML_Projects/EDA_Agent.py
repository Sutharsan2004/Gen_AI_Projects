from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from google.colab import userdata
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

#API key assigning
api = userdata.get('GROQ_API_KEY')

def analyze_dataset(file_path, api):
    df = pd.read_csv(file_path)
    
    llm = ChatGroq(
        temperature=0, 
        model_name="llama-3.3-70b-versatile",
        api_key=api
    )
    
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True,
        allow_dangerous_code=True,
        handle_parsing_errors=True  
    )
    
    
    query = """
    Perform a full EDA. Specifically:
    1. Check for nulls.
    2. Check datatypes.
    3. Analyze the distribution of the target variable.
    4. Check for correlations.
    5. Create Visualizations  
    Finally, output a detailed TEXT SUMMARY of your findings.
    """
    
   # invoking the model
    response = agent.invoke({"input": query})
    
   # Final answer
    return response['output']


def ask_llm(eda_summary, api):
    
    if not eda_summary:
        print("No EDA results found.")
        return

    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0.7,
        api_key=api
    )

    # prompt for the model
    template = """
    You are a helpful AI Data Science assistant.
    
    I have performed an EDA on a dataset and here are the findings:
    
    --- EDA FINDINGS ---
    {eda}
    --------------------
    
    Based on these findings, answer the user's question.
    If the user asks what model to build, recommend algorithms that fit this specific data distribution (e.g., if data is imbalanced, suggest XGBoost/SMOTE).
    
    User Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    print("\nEDA Complete. Chat with your Data")
    print("Enter 'quit' or 'exit' to stop.\n")
    
    while True:
        user_query = input("You: ")
        if user_query.lower() in ['quit', 'exit']:
            break
        
        # Invoke the chain
        response = chain.invoke({"eda": eda_summary, "question": user_query})
        print(f"AI: {response.content}\n")


file_path = "" # your dataset file path

# EDA Agent
print("Running EDA Agent...")
eda_text_result = analyze_dataset(file_path, api)

# Summary model
ask_llm(eda_text_result, api)
