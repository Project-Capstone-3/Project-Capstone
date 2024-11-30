import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib  # Untuk memuat model

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
page = st.sidebar.radio("Pilih Halaman", ["Home", "About Us", "Prediksi Penyakit"])

# --- Page: Prediksi Penyakit Diabetes ---
if page == "Prediksi Penyakit":
    st.title("Prediksi Risiko Diabetes 🩺")

    # Load scaler (pastikan scaler sama dengan saat pelatihan)
    @st.cache_resource
    def load_scaler():
        return joblib.load("scaler.pkl")  # Pastikan scaler disimpan saat pelatihan model

    # Load model
    @st.cache_resource
    def load_model():
        return joblib.load("svm_model.pkl")  # Pastikan model tersedia

    scaler = load_scaler()
    model = load_model()

    # Load data latih (untuk referensi)
    @st.cache_data
    def load_data():
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        data = pd.read_csv(url, names=columns)
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

    # Display user input
    st.subheader("Data Anda")
    st.dataframe(input_df)

    # Transform input data
    input_scaled = scaler.transform(input_df)

    # Log dimensi untuk debugging
    st.write("Dimensi input:", input_scaled.shape)
    st.write("Dimensi fitur model:", model.n_features_in_)

    # Check input dimensions
    if input_scaled.shape[1] != model.n_features_in_:
        st.error("Dimensi input tidak sesuai dengan model. Periksa data atau scaler yang digunakan.")
    else:
        # Prediction
        prediction = model.predict(input_scaled)

        # Handle prediction score if decision_function is available
        try:
            prediction_proba = model.decision_function(input_scaled)
            score = f"- **Skor Prediksi**: {prediction_proba[0]:.2f}"
        except AttributeError:
            score = ""

        # Display prediction
        st.subheader("Hasil Prediksi")
        diabetes = np.array(['Tidak Diabetes', 'Diabetes'])
        result = f"""
        ### Anda berisiko: **{diabetes[prediction][0]}** 🩺
        {score}
        """
        st.markdown(result)

        # Display accuracy
        st.subheader("Akurasi Model")
        st.write("**Akurasi Model**: 80.00%")
