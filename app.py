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
    with st.sidebar.form("input_form"):
        pregnancies = st.text_input("Kehamilan (Pregnancies)", "3")
        glucose = st.text_input("Glukosa (Glucose)", "120")
        blood_pressure = st.text_input("Tekanan Darah (BloodPressure)", "70")
        skin_thickness = st.text_input("Ketebalan Kulit (SkinThickness)", "20")
        insulin = st.text_input("Insulin", "79")
        bmi = st.text_input("BMI", "20.0")
        diabetes_pedigree_function = st.text_input("Fungsi Pedigree Diabetes", "0.47")
        age = st.text_input("Usia", "33")
        
        # Tombol submit
        submitted = st.form_submit_button("Prediksi")

    if submitted:
        # Validasi input
        try:
            input_data = {
                "Pregnancies": float(pregnancies),
                "Glucose": float(glucose),
                "BloodPressure": float(blood_pressure),
                "SkinThickness": float(skin_thickness),
                "Insulin": float(insulin),
                "BMI": float(bmi),
                "DiabetesPedigreeFunction": float(diabetes_pedigree_function),
                "Age": float(age),
            }
            input_df = pd.DataFrame([input_data])

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

        except ValueError:
            st.error("Mohon masukkan data numerik yang valid!")

