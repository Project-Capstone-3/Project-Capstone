import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="SugarGuard - Kenali Risikomu, Prediksi Diabetesmu",
    page_icon="https://github.com/Project-Capstone-3/Project-Capstone/blob/master/SugarGuard.png?raw=true",
    layout="wide",
)

# --- Fungsi Memuat Model, Scaler, dan PCA ---
@st.cache_resource
def load_model_scaler_pca():
    # Muat model, scaler, dan PCA
    model = joblib.load("svm_model.pkl")  # Sesuaikan nama file model
    scaler = joblib.load("minmax_scaler.pkl")  # Muat scaler MinMaxScaler yang telah disimpan
    pca = joblib.load("pca_model.pkl")  # Muat model PCA yang telah disimpan
    return model, scaler, pca

model, scaler, pca = load_model_scaler_pca()

# --- Sidebar ---
st.sidebar.image(
    "https://github.com/Project-Capstone-3/Project-Capstone/blob/master/SugarGuard.png?raw=true",
)
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Home", "About Us", "Prediksi Penyakit"])

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
    - **Instagram**: [@SugarGuard](https://www.instagram.com/sugardguard/profilecard/?igsh=bWp0N2ZiNHF1andj)
    """
    )

# --- Page: Prediksi Penyakit Diabetes ---
elif page == "Prediksi Penyakit":
    st.title("Prediksi Risiko Diabetes 🩺")

    # Input data pengguna dengan input angka, bukan slider
    pregnancies = st.number_input("Kehamilan (Pregnancies)", min_value=0, max_value=17, value=3)
    glucose = st.number_input("Glukosa (Glucose)", min_value=0, max_value=200, value=120)
    blood_pressure = st.number_input("Tekanan Darah (BloodPressure)", min_value=0, max_value=122, value=70)
    skin_thickness = st.number_input("Ketebalan Kulit (SkinThickness)", min_value=0, max_value=99, value=20)
    insulin = st.number_input("Insulin", min_value=0.0, max_value=846.0, value=79.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=67.1, value=20.0)
    diabetes_pedigree_function = st.number_input("Fungsi Keturunan Diabetes", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Usia", min_value=21, max_value=81, value=33)

    # Menyusun input data ke dalam DataFrame
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree_function],
        "Age": [age],
    })

    # Tampilkan data input
    st.subheader("Data Anda")
    st.dataframe(input_data)

    # Skalakan data input menggunakan MinMaxScaler yang telah dilatih
    try:
        input_scaled = scaler.transform(input_data)

        # Transformasikan data input dengan PCA
        input_pca = pca.transform(input_scaled)

        # Prediksi menggunakan model
        prediction = model.predict(input_pca)
        prediction_proba = model.predict_proba(input_pca)

        diabetes_labels = ['Tidak Diabetes', 'Diabetes']

        st.subheader("Hasil Prediksi")
        st.markdown(f"""
        ### Anda berisiko: **{diabetes_labels[prediction[0]]}** 🩺
        - **Probabilitas Tidak Diabetes**: {prediction_proba[0][0]*100:.2f}%  
        - **Probabilitas Diabetes**: {prediction_proba[0][1]*100:.2f}%
        """)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
