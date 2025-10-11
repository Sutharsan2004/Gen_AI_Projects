import streamlit as st
import string

def clean_data(text):
    text = text.lower()
    for punct in string.punctuation:
        text=text.replace(punct, "")
    return text

def frequency_count(text):
    words = text.split()
    counts={}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts

st.title("Hello Streamlit!")
st.write("Enter your text below!")

user_input = st.text_area("Enter your text here")

if st.button("Analyse"):
    clean = clean_data(user_input)
    word = frequency_count(clean)
    sorted_words=sorted(word.items(), key= lambda x: x[1], reverse=True)
    top=sorted_words[:3]

    st.subheader("Text preprocessing")
    st.write(clean)

    st.subheader("Word Frequecy Count")
    st.write(word)

    st.subheader("Top 3 word frequencies")
    st.write(top)

    st.bar_chart(dict(sorted_words))