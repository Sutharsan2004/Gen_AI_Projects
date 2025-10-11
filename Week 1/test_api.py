import requests
from flask import Flask, jsonify

app = Flask(__name__)

url = "http://127.0.0.1:5000/process"
head = {"Content_Type" : "application/JSON"}
data = {"text": "Generative AI is awesome! AI is the future. AI rocks."}


@app.route("/home")
def home():
   

    response = requests.post(url, headers=head, json=data)

    return jsonify(response.json())

if __name__ == "__main__":
    app.run(port=5001, debug=True)