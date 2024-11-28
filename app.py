import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from io import StringIO

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="SugarGuard - Kenali Risikomu, Prediksi Diabetesmu",
    page_icon="https://github.com/Project-Capstone-3/Project-Capstone/blob/master/SugarGuard.png?raw=true",
    layout="wide",
)

# --- Sidebar ---
st.sidebar.image(
    "https://github.com/Project-Capstone-3/Project-Capstone/blob/master/SugarGuard.png?raw=true",
)
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Home", "About Us", "Prediksi Penyakit Diabetes"])

# --- Page: Home ---
if page == "Home":
    st.title("Selamat Datang di SugarGuard!")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://github.com/Project-Capstone-3/Project-Capstone/blob/master/SugarGuard.png?raw=true", use_column_width=True)
    with col2:
        st.markdown("""
        ### Apa itu SugarGuard?
        SugarGuard adalah aplikasi berbasis **Machine Learning** yang dirancang untuk membantu Anda
        memprediksi risiko terkena diabetes. Kami menyediakan alat yang mudah digunakan dan
        akurat untuk membantu pengambilan keputusan kesehatan Anda.

        ### Fitur Utama:
        - **Prediksi Risiko Diabetes** dengan hasil akurat.
        - **Visualisasi Data** untuk memahami kesehatan Anda.
        - **Akurasi Tinggi** dari model Machine Learning terbaik.

        ** Coba sekarang dan pastikan kesehatan Anda!**
        """)

# --- Page: About Us ---
elif page == "About Us":
    st.title("Tentang Kami 👩‍💻👨‍💻")
    st.markdown("""
    Kami adalah tim ahli yang memiliki tujuan untuk menggabungkan **teknologi** dan **kesehatan** 
    demi masa depan yang lebih baik. Dengan menggunakan algoritma pembelajaran mesin, kami ingin
    membantu lebih banyak orang memahami risiko kesehatan mereka.

    ### Kontak Kami
    - **Email**: sugarguard@support.com
    - **Instagram**: [@SugarGuard](https://instagram.com)
    """
    )

# --- Page: Prediksi Penyakit Diabetes ---
elif page == "Prediksi Penyakit Diabetes":
    st.title("Prediksi Risiko Diabetes 🩺")

    # Fungsi untuk memuat model SVM dari GitHub
    @st.cache_resource
    def load_model():
        model_url = 'https://github.com/Project-Capstone-3/Project-Capstone/raw/master/svm_model.pkl'
        response = requests.get(model_url)
        model = joblib.load(StringIO(response.text))  # Load the model from the downloaded content
        return model

    svm_model = load_model()

    # Fungsi untuk memuat data lokal dari GitHub CSV
    @st.cache_data
    def load_data():
        data_url = 'https://github.com/Project-Capstone-3/Project-Capstone/raw/master/diabetes.csv'
        response = requests.get(data_url)
        data = pd.read_csv(StringIO(response.text))
        return data

    data = load_data()

    st.sidebar.header("Masukkan Data Anda")
    input_df = pd.DataFrame({
        "Pregnancies": [st.sidebar.slider("Kehamilan (Pregnancies)", 0, 17, 3)],
        "Glucose": [st.sidebar.slider("Glukosa (Glucose)", 0, 200, 120)],
        "BloodPressure": [st.sidebar.slider("Tekanan Darah (BloodPressure)", 0, 122, 70)],
        "SkinThickness": [st.sidebar.slider("Ketebalan Kulit (SkinThickness)", 0, 99, 20)],
        "Insulin": [st.sidebar.slider("Insulin", 0.0, 846.0, 79.0)],
        "BMI": [st.sidebar.slider("BMI", 0.0, 67.1, 20.0)],
        "DiabetesPedigreeFunction": [st.sidebar.slider("Fungsi Pedigree Diabetes", 0.0, 2.42, 0.47)],
        "Age": [st.sidebar.slider("Usia", 21, 81, 33)],
    })

    # Tampilkan data input pengguna
    st.subheader("Data Anda")
    st.dataframe(input_df)

    # Preprocessing data
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Prediksi menggunakan model SVM
    input_scaled = scaler.transform(input_df)
    prediction = svm_model.predict(input_scaled)
    prediction_proba = svm_model.predict_proba(input_scaled)

    # Tampilkan hasil prediksi
    st.subheader("Hasil Prediksi")
    diabetes = np.array(['Tidak Diabetes', 'Diabetes'])
    result = f"""
    ### Anda berisiko: **{diabetes[prediction][0]}** 🩺
    - **Probabilitas Tidak Diabetes**: {prediction_proba[0][0]*100:.2f}%
    - **Probabilitas Diabetes**: {prediction_proba[0][1]*100:.2f}%
    """
    st.markdown(result)

    # Tampilkan akurasi model
    st.subheader("Akurasi Model")
    accuracy = accuracy_score(y_test, svm_model.predict(X_test))
    st.write(f"**Akurasi Model**: {accuracy*100:.2f}%")

    # Tambahkan elemen interaktif: Bar Chart untuk distribusi data
    st.subheader('Distribusi Data Pengguna Dibandingkan Data Latihan')
    chart_data = pd.DataFrame({
        "Pengguna": input_df.iloc[0],
        "Rata-rata Data Latihan": data.mean()
    }).T

    st.bar_chart(chart_data)
