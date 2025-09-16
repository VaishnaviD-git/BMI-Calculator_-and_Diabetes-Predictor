
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor

st.set_page_config(page_title="Health Analyzer", layout="centered")
st.title("🩺 Health Prediction & Self-Analysis Tool")
st.write("This application allows users to analyze their health by predicting BMI and diabetes risk based on simple medical information.")

# Sidebar menu
option = st.sidebar.selectbox("Choose a prediction tool", (
    "BMI Prediction - Linear Regression",
    "BMI Category - Logistic Regression",
    "Diabetes Prediction - Naive Bayes",
    "BMI Prediction - KNN",
    "Visualizations"
))

# Linear Regression for BMI
if option == "BMI Prediction - Linear Regression":
    df = pd.read_csv("with_bmi.csv").dropna()
    le_gender = LabelEncoder()
    df['Gender_encoded'] = le_gender.fit_transform(df['Gender'])
    X = df[['Weight', 'Height', 'Gender_encoded']]
    y = df['BMI']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    st.write(f"**Model R² Score:** {r2:.4f}")

    user_weight = st.number_input("Enter weight (kg)", min_value=0.0)
    user_height = st.number_input("Enter height (m)", min_value=0.0)
    user_gender = st.selectbox("Select gender", le_gender.classes_)
    if st.button("Predict BMI"):
        gender_encoded = le_gender.transform([user_gender])[0]
        predicted_bmi = model.predict([[user_weight, user_height, gender_encoded]])[0]
        st.success(f"Predicted BMI: {predicted_bmi:.2f}")

# Logistic Regression for BMI Category
elif option == "BMI Category - Logistic Regression":
    df = pd.read_csv("with_bmi.csv").dropna()
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    X = df[['Weight', 'Height', 'Gender']]
    y = df['Index']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    st.write(f"**Accuracy:** {acc:.4f}")

    bmi_categories = {
        0: "Extremely Weak", 1: "Weak", 2: "Normal", 3: "Overweight", 4: "Obesity"
    }
    user_weight = st.number_input("Weight (kg)", key="w2")
    user_height = st.number_input("Height (m)", key="h2")
    user_gender = st.selectbox("Gender", ["Male", "Female"], key="g2")
    if st.button("Predict BMI Category"):
        gender_val = 0 if user_gender == "Male" else 1
        prediction = model.predict([[user_weight, user_height, gender_val]])[0]
        st.success(f"BMI Category: {bmi_categories.get(prediction, 'Unknown')}")

# Naive Bayes for Diabetes
elif option == "Diabetes Prediction - Naive Bayes":
    df = pd.read_csv("diabetes_prediction_dataset.csv").dropna()
    label_encoders = {}
    for col in ['gender', 'smoking_history']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop(columns=['diabetes', 'hypertension', 'heart_disease'])
    y = df['diabetes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    st.write(f"**Accuracy:** {accuracy_score(y_test, model.predict(X_test)):.4f}")

    age = st.slider("Age", 1, 120)
    bmi = st.number_input("BMI", min_value=0.0)
    hba1c = st.number_input("HbA1c level", min_value=0.0)
    glucose = st.number_input("Blood Glucose level", min_value=0.0)
    gender_input = st.selectbox("Gender", label_encoders['gender'].classes_)
    smoking_input = st.selectbox("Smoking History", label_encoders['smoking_history'].classes_)
    if st.button("Predict Diabetes"):
        gender_enc = label_encoders['gender'].transform([gender_input])[0]
        smoking_enc = label_encoders['smoking_history'].transform([smoking_input])[0]
        input_data = [[gender_enc, age, smoking_enc, bmi, hba1c, glucose]]
        result = model.predict(input_data)[0]
        st.success("Prediction: Diabetic" if result == 1 else "Prediction: Non-Diabetic")

# KNN Regression for BMI
elif option == "BMI Prediction - KNN":
    df = pd.read_csv("with_bmi.csv").dropna()
    le_gender = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    X = df[['Height', 'Weight', 'Gender']]
    y = df['BMI']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(X_train, y_train)
    r2 = r2_score(y_test, knn.predict(X_test))
    rmse = mean_squared_error(y_test, knn.predict(X_test)) ** 0.5
    st.write(f"**KNN R² Score:** {r2:.4f}, RMSE: {rmse:.2f}")

    user_height = st.number_input("Height (m)", key="h3")
    user_weight = st.number_input("Weight (kg)", key="w3")
    user_gender = st.selectbox("Gender", le_gender.classes_, key="g3")
    if st.button("Predict using KNN"):
        gender_encoded = le_gender.transform([user_gender])[0]
        prediction = knn.predict([[user_height, user_weight, gender_encoded]])[0]
        st.success(f"KNN Predicted BMI: {prediction:.2f}")

# Visualizations
elif option == "Visualizations":
    st.subheader("Data Visualizations")
    vis_type = st.selectbox("Choose visualization type", ["Diabetes Dataset", "BMI Dataset"])
    
    if vis_type == "Diabetes Dataset":
        df = pd.read_csv("diabetes_prediction_dataset.csv")
        le_gender = LabelEncoder()
        le_smoking = LabelEncoder()
        df['gender_encoded'] = le_gender.fit_transform(df['gender'])
        df['smoking_encoded'] = le_smoking.fit_transform(df['smoking_history'])
        features = {
            'age': 'Age',
            'smoking_encoded': 'Smoking History',
            'HbA1c_level': 'HbA1c Level',
            'blood_glucose_level': 'Blood Glucose Level',
            'bmi': 'BMI'
        }
        for col, label in features.items():
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=col, y='diabetes', hue='diabetes', palette='Set1', alpha=0.5, ax=ax)
            ax.set_title(f"{label} vs Diabetes")
            st.pyplot(fig)

    elif vis_type == "BMI Dataset":
        df = pd.read_csv("with_bmi.csv")
        df = df[['Height', 'Weight', 'Gender', 'BMI']].dropna()
        
        fig1, ax1 = plt.subplots()
        sns.scatterplot(x=df['Height'], y=df['BMI'], ax=ax1)
        ax1.set_title('Height vs BMI')
        st.pyplot(fig1)
        
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x=df['Weight'], y=df['BMI'], ax=ax2)
        ax2.set_title('Weight vs BMI')
        st.pyplot(fig2)
