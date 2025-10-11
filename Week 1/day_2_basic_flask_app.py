from flask import Flask

# app=Flask(__name__)

# @app.route('/')
# def home():
#     return "Hello World!!"

# @app.route('/about')
# def abt():
#     return "I'm dharshan studying AIDS dept"



# if __name__=='__main__':
#     app.run(debug=True)

# app.py
from flask import Flask, request, jsonify
import string


app = Flask(__name__)

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

# @app.route("/")
# def home():
#     return "Flask API Testing"

@app.route("/process", methods=["POST"])
def process_text():
    data = request.get_json()
    text = data.get("text", "")
    clean = clean_data(text)
    word = frequency_count(clean)
    sorter_words = sorted( word.items(), key= lambda x:x[1], reverse=True)
    top = sorter_words[:3]

    return jsonify({"Cleaned":clean, "Top" : top})

if __name__ == "__main__":
    app.run(debug=True)
