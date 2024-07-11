import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load the pretrained KMeans model
with open('models/kmeans_model.pkl', 'rb') as file:
    kmeans_model = pickle.load(file)
with open("models/scaler.pkl","rb") as file:
    scaler = pickle.load(file)

def get_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text

def preprocess_and_predict(age,gender,salary,spending_score):
    # Standardize the data (assuming your model was trained on standardized data)
    data = pd.DataFrame({
        "Annual Income (k$)":[salary],
        "Spending Score (1-100)":[spending_score]
              })
    data_scaled = scaler.transform(data)

    # Predict the cluster
    cluster = kmeans_model.predict(data_scaled)
    if cluster[0] == 0:
        cust_type = "Mid-Income, Mid-Spenders"
        prompt =  f"""As a senior marketing expert and owner of the mall, devise mid-tier marketing strategies for customers of age {age} and gender {gender} with moderate spending scores and 
annual income. Emphasize the quality and value of products, personalized shopping experiences, and seasonal offers that encourage
steady spending. Specifically, design campaigns that offer personalized product recommendations based on past purchases, 
create loyalty programs that provide increasing rewards with continued spending, and host community events that make these 
customers feel valued and connected to the brand.The output should be in bullet points indicating the marketing strategies"""
    elif cluster[0] == 1:
        cust_type = "High-Income, Low Spenders"
        prompt = f"""As a senior marketing expert and owner of the mall, propose innovative strategies to increase spending among high-income customers of age {age} and gender {gender} with 
low spending scores. Focus on personalized recommendations, premium product showcases, and targeted campaigns that 
highlight the potential benefits and value of increased spending. Develop high-impact strategies such as 
curated shopping experiences tailored to their preferences, exclusive previews of high-end products, 
and targeted communication that emphasizes the unique benefits and superior quality of products available 
to them. How can we leverage their high income to convert potential into actual expenditure, ensuring they feel the exclusivity 
and value in their purchases without feeling pressured?The output should be in bullet points indicating the marketing strategies"""
    elif cluster[0] == 2:
        cust_type = " Low-Income, Low Spenders"
        prompt = f""" As a senior marketing expert and owner of the mall, suggest budget-friendly promotions and discounts that can attract and engage customers of age {age} and gender {gender} 
with low spending scores and low annual income. Consider strategies that emphasize value for money and essential purchases, 
such as bundle deals, loyalty rewards for frequent purchases, and limited-time offers on essential items. The output should be in bullet points indicating the marketing strategies"""
    elif cluster[0] == 3:
        cust_type = "Low-Income, High Spenders"
        prompt = f"""As a senior marketing expert and owner of the mall, recommend strategies to optimize spending habits of customers of age {age} and gender {gender} with high spending scores 
but low annual income. Focus on loyalty programs, exclusive member discounts, and installment payment options to maintain their 
spending while ensuring affordability. Additionally, consider creating special savings events that provide significant value and 
appeal to their desire to spend.The output should be in bullet points indicating the marketing strategies"""
    elif cluster[0] == 4:
        cust_type = "High-Income, High Spenders"
        prompt = f"""As a senior marketing expert and owner of the mall, develop premium marketing strategies for high-spending, high-income customers of age {age} and gender {gender}. 
Highlight luxury products, exclusive events, VIP experiences, and personalized services. How can we create a sense of exclusivity 
and cater to their high standards to foster brand loyalty? 
Consider implementing white-glove services, personal shopping assistants, and early access to new collections or limited-edition 
items.The output should be in bullet points indicating the marketing strategies"""
    return cluster[0],cust_type,prompt

# Define the main function to run the app
def main():
    # Custom CSS for styling
    st.markdown("""
        <style>
            .main {
                background-color: #f0f0f0;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #231c1c;
                text-align: center;
                margin-bottom: 20px;
            }
            .stButton button {
                background-color: #231c1c;
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 1em;
            }
            .stButton button:hover {
                background-color: #f0f8ff;
            }
            .input-container {
                margin-bottom: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main">', unsafe_allow_html=True)
    
    st.title("ShopperProfiler")
    st.subheader("""
        Unlock customer segments and refine marketing strategies effortlessly.
    """)
    st.write("""This app provides insights into customer segments based on age, gender, annual income, and spending score. 
             It helps you understand your customers better and tailor marketing strategies accordingly""")

    # Input fields with better layout
    st.header("Customer Details")
    age = st.number_input("Enter Age", min_value=0, max_value=120, value=30)
    gender = st.radio("Gender", ["Male", "Female"], help="Select the gender of the customer")
    annual_income = st.number_input("Annual Income (in k$)", min_value=1, value=300, help="Enter the annual income of the customer in thousands of dollars")
    spending_score = st.number_input("Enter Spending Score (1-100)", min_value=1, max_value=100, value=50)
    
    columns = st.columns((2, 1, 2))

    if columns[1].button("Submit"):
        # Preprocess and predict
        cluster, cust_type, prompt = preprocess_and_predict(age, gender,annual_income, spending_score)
        response = get_gemini_response(prompt)

        # Display the result
        st.success(f"The customer belongs to cluster: {cluster + 1}")
        st.info(f"Type of customer: {cust_type}")
        st.write("Here are some strategies for effective marketing for this customer type:")
        st.write(response)
    
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == '__main__':
    main()

