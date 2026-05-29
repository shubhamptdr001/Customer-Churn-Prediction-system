# 📊 Customer Churn Prediction System

A machine learning web application that predicts whether a telecom customer is likely to churn (leave the service). Built with **Logistic Regression** and deployed using **Streamlit** for an interactive user experience.

---

## 🎯 Overview

Customer churn is a critical metric for telecom companies. This project uses historical customer data from the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset on Kaggle to train a Logistic Regression model that predicts churn risk and provides actionable retention tips.

---

## 🏗️ Project Structure

```
Customer-Churn-Prediction-system/
│
├── Main1.ipynb        # Jupyter notebook — data loading, preprocessing, model training & evaluation
├── app1.py            # Streamlit web app for real-time churn prediction
├── model.pkl          # Trained Logistic Regression model
├── scaler.pkl         # Fitted StandardScaler for feature scaling
├── encoders.pkl       # Fitted LabelEncoders for categorical features
├── Churn1.pkl         # Alternate saved model (initial version)
└── README.md          # Project documentation
```

---

## 📋 Features Used

| Feature        | Type        | Description                                      |
|----------------|-------------|--------------------------------------------------|
| Gender         | Categorical | Male / Female                                    |
| SeniorCitizen  | Binary      | Whether the customer is a senior citizen (Yes/No)|
| Partner        | Categorical | Whether the customer has a partner (Yes/No)      |
| Dependents     | Categorical | Whether the customer has dependents (Yes/No)     |
| Tenure         | Numerical   | Number of months the customer has stayed          |
| PhoneService   | Categorical | Whether the customer has phone service (Yes/No)  |
| MultipleLines  | Categorical | Yes / No / No phone service                      |
| Contract       | Categorical | Month-to-month / One year / Two year             |
| TotalCharges   | Numerical   | Total amount charged to the customer              |

---

## 🧠 Model Details

- **Algorithm**: Logistic Regression (`C=0.1`, `solver='liblinear'`)
- **Preprocessing**: Label Encoding for categorical features + Standard Scaling
- **Train/Test Split**: 80/20 with `random_state=42`
- **Accuracy**: ~77.7%

### Classification Report

```
              precision    recall  f1-score   support

           0       0.80      0.92      0.86      1036
           1       0.63      0.38      0.47       373

    accuracy                           0.78      1409
   macro avg       0.72      0.65      0.67      1409
weighted avg       0.76      0.78      0.76      1409
```

---

## 🚀 How to Run

### Prerequisites

- **Python 3.9+** installed on your system
- **pip** (Python package manager)

### 1. Clone the Repository

```bash
git clone https://github.com/shubhamptdr001/Customer-Churn-Prediction-system.git
cd Customer-Churn-Prediction-system
```

### 2. Install Dependencies

```bash
pip install streamlit pandas scikit-learn kagglehub
```

### 3. (Optional) Retrain the Model

If you want to retrain the model from scratch, open and run the Jupyter notebook:

```bash
jupyter notebook Main1.ipynb
```

> This will download the dataset from Kaggle via `kagglehub`, train the model, and save `model.pkl`, `scaler.pkl`, and `encoders.pkl`.

### 4. Launch the Streamlit App

```bash
streamlit run app1.py
```

This will start a local development server and automatically open the app in your default browser at:

```
http://localhost:8501
```

### 5. Use the Application

1. Fill in the customer details using the dropdown menus and input fields:
   - Gender, Senior Citizen, Partner, Dependents
   - Tenure (months), Phone Service, Multiple Lines
   - Contract Type, Total Charges
2. Click the **"Predict"** button
3. View the prediction result:
   - 🟢 **Low Risk of Churn** — with confidence percentage and retention tips
   - 🔴 **High Risk of Churn** — with confidence percentage and prevention strategies

---

## 🛠️ Tech Stack

| Component       | Technology                  |
|-----------------|-----------------------------|
| Language        | Python 3.12                 |
| ML Framework    | scikit-learn                |
| Web Framework   | Streamlit                   |
| Data Handling   | pandas                      |
| Dataset Source  | Kaggle (via kagglehub)      |
| Serialization   | pickle                      |

---

## 📸 Screenshots

### Input Form
The app provides an intuitive dark-themed form to enter customer attributes like gender, tenure, contract type, and charges.

### Prediction Result
After clicking **Predict**, the app displays:
- A color-coded risk indicator (green for low risk, red for high risk)
- Churn probability percentage
- Actionable tips for customer retention or churn prevention

---

## 📄 License

This project is open source and available for educational and personal use.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

> **Built with ❤️ using Python, scikit-learn, and Streamlit**
