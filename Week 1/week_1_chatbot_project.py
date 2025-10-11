# Week one project Simple Chatbot
import requests
import streamlit as st

url = "https://apifreellm.com/api/chat"
headers = {"Content-Type" : "aplication/json"}

if "history" not in st.session_state:  #made a misyake history nor History
    st.session_state.history=[]

st.write("Simple Chatbot")
user_input = st.text_input("You :", placeholder="Enter your prompt here")

if st.button("Ask!") and user_input:
    data = {"message" : user_input}  # message is the correct parameter not Message
    with st.spinner("Generating Response..."):
        response = requests.post(url, headers=headers, json=data)   #url is enough no need for url=url. json is mandatory don't use data instead
    if response.status_code == 200:
        js = response.json()
        if js.get("status") == "success":
            ai_resp = js.get("response")
            st.session_state.history.append((user_input, ai_resp))
        #st.write(ai_resp)
    else:
        st.write("Some error occured")

for i, (q,a) in enumerate(st.session_state.history): #if enumerate then use i, (a,b), if it is reveresed use a,b
    st.markdown(f"**You:** {q}")
    st.markdown(f"**AI:** {a}")    