# import requests

# # response = requests.get("https://api.github.com")
# # print(response.status_code)  # 200 means success
# # #print(response.json())

# API_URL = "https://api-inference.huggingface.co/models/gpt2"
# headers = {}
# prompt = "What is gen ai"
# response = requests.post(API_URL, headers=headers, json={"Input":prompt})

# print(response)
import requests
import streamlit as st

url = "https://apifreellm.com/api/chat"
headers = {
    "Content-Type": "application/json"
}
st.write("Hello Dharsh!!!")
msg = st.text_area("Enter promt")
inp = str(msg)

if "history" not in st.session_state:
    st.session_state.history = []


if st.button("Ask!"):
    data = {
        "message": inp
    }
    with st.spinner("Generating Response...."):
        response = requests.post(url, headers=headers, json=data)


    if response.status_code == 200:
        js = response.json()
        if js.get("status") == "success":
            #st.write("LLM response:", js.get("response"))
            st.session_state.history.append((inp, js.get("response")))
        else:
            st.write("Error from API:", js.get("error"), js.get("status"))
    else:
        st.write("HTTP Error:", response.status_code, response.text)

for i, (q, a) in enumerate(st.session_state.history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**AI:** {a}")





