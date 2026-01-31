import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler

label_encoder = LabelEncoder()
scaler = StandardScaler()
model = pickle.load(open('Churn.pkl','rb'))

st.title("Churn Prediction and Suggestion System")
gender = st.selectbox('Select Gender:',options=['Male','Female'])
Sc = st.selectbox('Select Citizenship:',options=['Yes','No'])
Partner = st.selectbox('Select Partner:',options=['Yes','No'])
Dependents = st.selectbox('Select Dependents:',options=['Yes','No'])
tenure = st.text_input("Enter your tenure:")
PhoneServices = st.selectbox('Select PhoneServices:',options=['Yes','No'])
MultipleLines = st.selectbox('Select MultipleLines:',options=['Yes','No','No phone services'])
contract = st.selectbox('Select contract:',options=['One Year','Two year','Month-to-Month'])
TotalCharges = st.text_input("Enter your Total charges:")


def predictive(gender,Sc,Partner,Dependents,tenure,PhoneServices,MultipleLines,contract,TotalCharges):
    data = {
    "gender":[gender],
    "SeniorCitizen":[Sc],
    "Partner" :[Partner],
    "Dependents":[Dependents],
    "tenure":[tenure],
    "PhoneService":[PhoneServices],
    "MultipleLines":[MultipleLines],
    "Contract":[contract],
    "TotalCharges":[TotalCharges]
    }
    df1 = pd.DataFrame(data) 

    categorical_columns = ['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','Contract','TotalCharges']
    for column in categorical_columns:
        df1[column] = label_encoder.fit_transform(df1[column])

    df1 = scaler.fit_transform(df1)
    result = model.predict(df1).reshape(1,-1)
    return result[0]

#Tips for Churn
churn_tips_data = {
    "Tips for Churn Prevention": [
        "Identify the Reasons: Understand why customers or employees are leaving. Conduct surveys, interviews, or exit interviews to gather feedback and identify common issues or pain points.",
        "Improve Communication: Maintain open and transparent communication channels. Address concerns promptly and proactively. Make sure customers or employees feel heard and valued.",
        "Enhance Customer/Employee Experience: Focus on improving the overall experience. This could involve improving product/service quality or creating a more positive work environment for employees.",
        "Offer Incentives: Provide incentives or loyalty programs to retain customers. For employees, consider benefits, bonuses, or career development opportunities.",
        "Personalize Interactions: Tailor interactions and offers to individual needs and preferences. Personalization can make customers or employees feel more connected and valued.",
        "Monitor Engagement: Continuously track customer or employee engagement. For customers, this might involve monitoring product usage or website/app activity. For employees, assess job satisfaction and engagement levels.",
        "Predictive Analytics: Use data and predictive analytics to anticipate churn. Machine learning models can help identify patterns and predict which customers or employees are most likely to churn.",
        "Feedback Loop: Create a feedback loop for ongoing improvement. Regularly seek feedback, analyze it, and use it to make informed decisions and changes.",
        "Employee Training and Development: Invest in training and development programs for employees. Opportunities for growth and skill development can improve job satisfaction and loyalty.",
        "Competitive Analysis: Stay aware of what competitors are offering. Ensure your products, services, and workplace environment remain competitive in the market."
    ]
}

# Tips for Customer Retention (Not leaving the company services)
retention_tips_data = {
    "Tips for Customer Retention": [
        "Provide Exceptional Customer Service: Ensure that customers receive excellent customer service and support.",
        "Create Loyalty Programs: Reward loyal customers with discounts, special offers, or exclusive access to products/services.",
        "Regularly Communicate with Customers: Keep customers informed about updates, new features, and promotions.",
        "Offer High-Quality Products/Services: Consistently deliver high-quality products or services that meet customer needs.",
        "Resolve Issues Quickly: Address customer concerns and issues promptly to maintain their satisfaction.",
        "Build Strong Customer Relationships: Develop strong relationships with customers by understanding their needs and preferences.",
        "Provide Value: Offer value-added services or content that keeps customers engaged and interested.",
        "Simplify Processes: Make it easy for customers to do business with you. Simplify processes and reduce friction.",
        "Stay Responsive: Be responsive to customer inquiries and feedback, even on social media and review platforms.",
        "Show Appreciation: Express gratitude to loyal customers and acknowledge their continued support."
    ]
}

churn_tips_df = pd.DataFrame(churn_tips_data)
retention_tips_df = pd.DataFrame(retention_tips_data)

if st.button("Predict"):
    result = predictive(gender,Sc,Partner,Dependents,tenure,PhoneServices,MultipleLines,contract,TotalCharges)

    if result == 0:
        st.title("Churn")
        st.write("Here are 10 tips for Churn Prevention:")
        st.dataframe(churn_tips_df, height=400,width=1000)
    else:
        st.title('Not Churn')
        st.write("Here are 10 tips for Customer Retention (Not Churning):")
        st.dataframe(retention_tips_df, height=400,width=1000)    