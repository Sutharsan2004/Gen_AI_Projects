import streamlit as st
import sqlite3
import pandas as pd
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# --- 1. SQL DATABASE LOGIC ---
DB_NAME = "crm_system.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS leads 
                 (name TEXT, project TEXT, email TEXT, budget INTEGER, priority TEXT, status TEXT)''')
    conn.commit()
    conn.close()

def execute_sql_workflow(data):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    intent = data.get('intent', 'log')
    original_name = data.get('name', 'Unknown')
    msg = f"Processing request for {original_name}..." # Default message
    
    if intent == 'delete':
        c.execute("DELETE FROM leads WHERE name LIKE ?", (f"%{original_name}%",))
        msg = f"🗑️ Deleted lead: {original_name}"
    
    elif intent == 'update':
        updates = []
        params = []
        
        # Dynamically check for ANY fields the AI provided
        # We check for 'new_name' specifically if user wants to rename someone
        target_name = data.get('new_name', original_name)
        
        fields_to_check = ['project', 'email', 'budget', 'status']
        for field in fields_to_check:
            if data.get(field):
                updates.append(f"{field} = ?")
                params.append(data[field])
                
                # Special Logic: If budget is updated, update priority too
                if field == 'budget':
                    updates.append("priority = ?")
                    params.append("High (VIP)" if int(data[field]) >= 50000 else "Standard")

        if data.get('new_name'):
            updates.append("name = ?")
            params.append(data['new_name'])

        if updates:
            query = f"UPDATE leads SET {', '.join(updates)} WHERE name LIKE ?"
            params.append(f"%{original_name}%")
            c.execute(query, params)
            msg = f"🔄 Updated records for {original_name}"
        else:
            msg = "⚠️ No specific changes identified to update."
            
    else: # Log/Insert
        budget = int(data.get('budget', 0))
        priority = "High (VIP)" if budget >= 50000 else "Standard"
        c.execute("INSERT INTO leads (name, project, email, budget, priority, status) VALUES (?, ?, ?, ?, ?, ?)", 
                  (original_name, data.get('project'), data.get('email'), budget, priority, "Active"))
        msg = f"✅ Added new lead: {original_name}"
    
    conn.commit()
    conn.close()
    return msg

# --- 2. UI & AI LOGIC ---
st.set_page_config(page_title="SQL Super CRM", layout="wide")
st.title("🧙‍♂️ Agentic SQL CRM Commander")

try:
    api_key = "your key"
except:
    api_key = None
    st.error("Add 'GROQ_API_KEY' to Colab Secrets.")

if api_key:
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
    init_db()

    prompt = ChatPromptTemplate.from_template(
        "You are a SQL CRM Agent. Extract details from: '{input}'.\n"
        "Intents: 'log', 'delete', 'update'.\n"
        "If renaming, put old name in 'name' and new name in 'new_name'.\n"
        "Return ONLY JSON with keys: name, new_name, project, email, budget, intent."
    )
    chain = prompt | llm | JsonOutputParser()

    cmd = st.text_input("Talk to your CRM:", placeholder="e.g., 'Change Sam's name to Samantha'")
    
    if st.button("Run Command"):
        if cmd:
            with st.spinner("Executing..."):
                structured_data = chain.invoke({"input": cmd})
                result_msg = execute_sql_workflow(structured_data)
                st.toast(result_msg)

    # --- 3. VIEW DATA ---
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM leads", conn)
    conn.close()
    st.dataframe(df, use_container_width=True)
