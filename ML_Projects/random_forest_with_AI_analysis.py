import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from google.colab import userdata
import os

api = userdata.get('GROQ_API_KEY')
os.environ['GROQ_API_KEY'] = api

data = {
    'Income_k':     [80,  90,  85,  100,   20,  25,  30,  22], # 4 Rich, 4 Poor
    'Credit_Score': [750, 500, 720, 480,   600, 600, 600, 600],
    'Has_CoSigner': [0,   0,   0,   0,     0,   1,   0,   1],
    'Loan_Status':  [1,   0,   1,   0,     0,   1,   0,   1]
}

df = pd.DataFrame(data)
X=df[['Income_k','Credit_Score','Has_CoSigner']]
y = df['Loan_Status']

params = {
    'n_estimators' : [50,100,150],
    'max_depth' : [3,5,7],
    'min_samples_split' :[2,3] # Removed the invalid value '1'
}

model = RandomForestClassifier(random_state=42)
grid_model = GridSearchCV(estimator=model, param_grid=params ,cv=2, scoring='accuracy' )
grid_model.fit(X, y) 

print(f"accuracy {grid_model.best_score_}")
print(f"parameters {grid_model.best_estimator_}")

best_model = grid_model.best_estimator_

inp = [[9, 750, 0]]
result = best_model.predict(inp)
print(f"Result :{"Approved" if result == 1 else 'Rejected'} Outcome : {result}")

def ask_model(x,y,inp,outp, api):
  llm = ChatGroq(
      model = "llama-3.1-8b-instant",
      temperature = 0.7,
      groq_api_key = api
  )

  template = """
  You are helpful AI assistent. You going to explain the following clearly.
  you will get a X, y, input and output of the machine learning model. Check the output and input and give the answer. If it is correct then explain why it is correct.
  if it is wrong then explain why it is wrong and what should i do.
  Finally give me the way to enhance the model outcome..
  I only need the answer for what i have asked, dont give unwanted answers.
  X : {x}
  y : {y}
  Input:{input}
  Output:{output}
  """

  prompt = ChatPromptTemplate.from_template(template)

  chain = prompt | llm

  response = chain.invoke({'x':X,'y':y, 'input':inp,'output':result})

  return response.content


def user_interaction(resp,query):
  llm = ChatGroq(
      model = "llama-3.1-8b-instant",
      temperature = 0.7,
      groq_api_key = api
  )

  template = """
  You are helpful AI assistant. You will receive a response of the AI model. Your job is to answer the questions based on that response.
  If question isn't related to response then say im not sure but,.. then your own response.
  Response : {response}
  Question:{question}

  """
  prompt = ChatPromptTemplate.from_template(template)

  chain = prompt | llm

  response = chain.invoke({'response':resp,'question':query})

  return response.content

while True:
  print("Ask If you have any query!!!")
  inp = input("You :\n")
  if inp.lower() == 'q':
    break
  res = ask_model(X, y,inp, result, api)
  print(f"AI outcome : {res}")
  user_interaction(res, inp)

  

