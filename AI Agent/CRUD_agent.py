
CSV_FILE = 'crm_data.csv'

def init_csv():
    if not os.path.exists(CSV_FILE):
        df = pd.DataFrame(columns=['name', 'project', 'email', 'budget', 'priority', 'status'])
        df.to_csv(CSV_FILE, index=False)

def save_lead(data):
    df = pd.read_csv(CSV_FILE)
    try:
        budget = int(data.get('budget', 0))
    except:
        budget = 0
    data['priority'] = "High (VIP)" if budget >= 50000 else "Standard"
    data['status'] = "Active"
    
    # Remove 'intent' so it doesn't save to CSV
    save_data = {k: v for k, v in data.items() if k != 'intent'}
    new_row = pd.DataFrame([save_data])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    return df

def delete_lead(name_to_delete):
    df = pd.read_csv(CSV_FILE)
    # Check if name exists (case-insensitive)
    if not df[df['name'].str.contains(name_to_delete, case=False, na=False)].empty:
        df = df[~df['name'].str.contains(name_to_delete, case=False, na=False)]
        df.to_csv(CSV_FILE, index=False)
        return True, df
    return False, df

# --- UI ---
st.set_page_config(page_title="Agentic CRM", layout="wide")
st.title("🚀 Agentic CRM: Create & Delete")

try:
    api_key = "Your_groq_api_key"
except:
    api_key = None
    st.error("Add 'GROQ_API_KEY' to Colab Secrets.")

if api_key:
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", groq_api_key=api_key)
    
    # UPDATED PROMPT: Now includes 'delete' intent
    prompt = ChatPromptTemplate.from_template(
        "You are a CRM Agent. Extract details from: '{input}'.\n"
        "Intents: 'log' (save), 'outreach' (save+email), 'delete' (remove lead).\n"
        "Return ONLY JSON: {{\"name\": \"...\", \"project\": \"...\", \"email\": \"...\", \"budget\": 0, \"intent\": \"...\"}}"
    )
    chain = prompt | llm | JsonOutputParser()

    cmd = st.text_input("Enter Command:", placeholder="e.g., 'Delete Sam' or 'Add Sutharsan budget 70k'")
    
    if st.button("Execute"):
        if cmd:
            init_csv()
            with st.spinner("Processing..."):
                structured_data = chain.invoke({"input": cmd})
                intent = structured_data.get('intent', 'log')
                name = structured_data.get('name')

                if intent == 'delete':
                    success, updated_df = delete_lead(name)
                    if success:
                        st.warning(f"🗑️ Deleted lead: {name}")
                    else:
                        st.error(f"Could not find lead named '{name}'")
                else:
                    updated_df = save_lead(structured_data)
                    st.success(f"✅ Added {name}")

                st.dataframe(updated_df, use_container_width=True)
