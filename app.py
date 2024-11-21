import streamlit as st
import pickle as pk
import os
import tensorflow 
from tensorflow.keras.models import load_model
import pandas as pd
import base64

def add_bg_image(image_file):
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    bg_image = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(bg_image, unsafe_allow_html=True)

# Add background image
add_bg_image("bg.png")

    
model = load_model(r"C:\Users\TANISHQ\Naresh_IT_Everyday_Personal\Artificial Intelligence\4. ANN\Churn Modelling project\ann_churn_model.h5")

st.title(' Customer Churn Predictor')

st.write("📌 The Customer Churn Predictor is a machine learning-powered web application designed to predict whether a customer is likely to churn (leave or cancel their service). Using an Artificial Neural Network (ANN), the model analyzes various customer data, including demographic information, financial details, and activity history, to predict the likelihood of churn.")

cities = ['Delhi','Banglore','Mumbai']
city = st.selectbox('Select city 🏠',cities)

genders = ['Male','Female']
gender = st.selectbox('Select Gender 👥',genders)

age = st.number_input('Enter Age 🎂',min_value=18)

yesno = ['Yes','No']
creditcard = st.selectbox('Owns a Credit Card 💳',yesno)

creditscore = st.number_input('Enter Credit Score 📊',min_value=300, max_value=900)

tenure = st.slider('Enter Tenure ⏳',min_value=0, max_value=15)

salary = st.number_input('Estimated Salary 💵')

balance = st.number_input('Enter Account Balance 🏦')

number_products = st.slider('Select Number of Products 🛒',min_value=0, max_value=10)

activemember = st.selectbox('Active Account 🔵',yesno)

#input
X_input = pd.DataFrame({
    'geo_banglore':[0],
    'geo_delhi':[0],
    'geo_mumbai':[0],
    'CreditScore':[creditscore],
    'gender':[gender],
    'age':[age],
    'tenure':[tenure],
    'balance':[balance],
    'NumOfProducts':[number_products],
    'HasCredCard':[creditcard],
    'ActiveMember':[activemember],
    'EstimatedSalary':[salary]
})


# One-hot encode the selected city
if city == 'Banglore':
    X_input['geo_banglore'] = 1
elif city == 'Delhi':
    X_input['geo_delhi'] = 1
elif city == 'Mumbai':
    X_input['geo_mumbai'] = 1


#Label encoded credit card , gender , Acitve member
if creditcard == 'Yes':
    X_input['HasCredCard'] = 1
else:
    X_input['HasCredCard']= 0


if gender == 'Male':
    X_input['gender'] = 1
else:
    X_input['gender']= 0
    

if activemember == 'Yes':
    X_input['ActiveMember']=1
else:
    X_input['ActiveMember']=0
    
  
# Convert to the desired format (a list of lists)
X_input_values = [[
    X_input['geo_banglore'][0],
    X_input['geo_delhi'][0],
    X_input['geo_mumbai'][0],
    X_input['CreditScore'][0],
    X_input['gender'][0],
    X_input['age'][0],
    X_input['tenure'][0],
    X_input['balance'][0],
    X_input['NumOfProducts'][0],
    X_input['HasCredCard'][0],
    X_input['ActiveMember'][0],
    X_input['EstimatedSalary'][0]
]]

# Convert to DataFrame
X_input_converted = pd.DataFrame(X_input_values, columns=X_input.columns)


import pickle 
sc = pickle.load(open(r"C:\Users\TANISHQ\Naresh_IT_Everyday_Personal\Artificial Intelligence\4. ANN\Churn Modelling project\scalar.pkl",'rb'))

new_customer_data = sc.transform(X_input_converted)

if st.button('Predict 🔮'):
    prediction = model.predict(new_customer_data)
    if prediction>= 0.5:
        st.success("✅ **Customer is not likely to churn.** 🟢")
    else:
        st.error("⚠️ **Customer is likely to churn!** ❌")
        
st.markdown("""
            ---
            <div style = "text-align: center;">
            Created by Tanishq Bololu <br>
            🚀 <a href="https://www.linkedin.com/in/tanishqbololu/" target="_blank">LinkedIn</a>
            </div>
            """, unsafe_allow_html=True)



    
# .\tensorflow_env\Scripts\activate