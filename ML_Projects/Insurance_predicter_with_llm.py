import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from google.colab import userdata
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Set up your API key here
api = userdata.get('GROQ_API_KEY')

# Simple dataset
data = {
    'age': [19, 28, 33, 32, 31, 46, 60, 25],
    'bmi': [27.9, 33.0, 22.7, 28.8, 25.7, 33.4, 35.1, 26.2],
    'children': [0, 3, 0, 0, 0, 1, 0, 1],
    'smoker': ['yes', 'no', 'no', 'no', 'no', 'no', 'yes', 'no'], # crucial feature
    'region': ['southwest', 'southeast', 'northwest', 'northwest', 'southeast', 'southeast', 'northeast', 'southwest'],
    'charges': [16884, 4449, 21984, 3866, 3756, 8240, 45000, 4200]
}

df = pd.DataFrame(data)

# Converting Yes/No to 1 and 0
df['smoker'] = df['smoker'].apply(lambda i: 1 if i == 'yes' else 0 )

# Handling regions in binary
df = pd.get_dummies(df, columns=['region'], drop_first=True)

X = df.drop("charges", axis = 1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_res = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
rf_res = rf.predict(X_test)

xg = XGBRegressor(n_estimators=100, learning_rate = 0.1)
xg.fit(X_train, y_train)
xg_res = xg.predict(X_test)

print(f"Linear Model Score in Training : {mean_absolute_error(y_test, lr_res):.2f}")
print(f"RF Model Score in Training : {mean_absolute_error(y_test, rf_res):.2f}")
print(f"XGB Model Score in Training : {mean_absolute_error(y_test, xg_res):.2f}")


def explain_premium(api, age, bmi, smoker, predicted_cost):
    
    llm = ChatGroq(
        model = "llama-3.1-8b-instant",
        groq_api_key = api,
        temperature=0.7
    )

    risk_level = "High" if predicted_cost > 10000 else "Low"
    
    template = f"""
    You are an Insurance Agent. You will receive the following details. 
    Client Details: 
    Age : {age}, 
    BMI : {bmi}, 
    Smoker: {smoker} {'Yes' if smoker==1 else 'No'}.
    AI Predicted Cost: {cost:.2f}
    
    Write a 100 words explanation for the client on WHY their premium is {risk_level}.
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm
    
    response = chain.invoke({"age":age, "bmi": bmi, "smoker":smoker, "cost":predicted_cost})

    return response.content


age = int(input("Enter your Age: "))
bmi = float(input("Enter BMI: "))
children = int(input("Enter No of Children: "))
smoker_input = input("Do you smoke? (yes/no): ").lower().strip()
region_input = input("Enter region (southwest/southeast/northwest/northeast): ").lower().strip()

smoker_val = 1 if smoker_input == 'yes' else 0

is_northwest = 1 if region_input == 'northwest' else 0
is_southeast = 1 if region_input == 'southeast' else 0
is_southwest = 1 if region_input == 'southwest' else 0

new_user = pd.DataFrame([[
    age, 
    bmi, 
    children, 
    smoker_val, 
    is_northwest, 
    is_southeast, 
    is_southwest
]], columns=X_train.columns)

cost = xg.predict(new_user)[0]

print(f"\n XGB Model prediction on new data: ${cost:,.2f}")
print("\n AI Explanation:")
explain_premium(api, age, bmi, smoker_val, cost)
