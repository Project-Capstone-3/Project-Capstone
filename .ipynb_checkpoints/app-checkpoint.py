import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

@st.cache
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(url, names=columns)
    return data

data = load_data()

st.sidebar.header('Masukkan Nilai Inputan Anda')

def user_input_features():
    Pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    Glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    BloodPressure = st.sidebar.slider('BloodPressure', 0, 122, 70)
    SkinThickness = st.sidebar.slider('SkinThickness', 0, 99, 20)
    Insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 79.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 20.0)
    DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction', 0.0, 2.42, 0.47)
    Age = st.sidebar.slider('Age', 21, 81, 33)

    data = {'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
            'Age': Age}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('Input Data Pengguna')
st.write(input_df)

X = data.drop('Outcome', axis=1)
y = data['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)
prediction_proba = model.predict_proba(input_scaled)

st.subheader('Prediksi')
diabetes = np.array(['Tidak Diabetes', 'Diabetes'])
st.write(diabetes[prediction])

st.subheader('Probabilitas Prediksi')
st.write(prediction_proba)

accuracy = accuracy_score(y_test, model.predict(X_test))
st.subheader('Akurasi Model')
st.write(f'{accuracy*100:.2f}%')