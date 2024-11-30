import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib  # Untuk memuat model

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="SugarGuard - Kenali Risikomu, Prediksi Diabetesmu",
    page_icon="https://github.com/Project-Capstone-3/Project-Capstone/blob/master/SugarGuard.png",
    layout="wide",
)

# --- Sidebar ---
st.sidebar.image(
    "https://github.com/Project-Capstone-3/Project-Capstone/blob/master/SugarGuard.png",
)
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["Home", "About Us", "Prediksi Penyakit"])

# --- Page: Home ---
if page == "Home":
    st.title("Selamat Datang di SugarGuard!")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://github.com/Project-Capstone-3/Project-Capstone/blob/master/SugarGuard.png", use_column_width=True)
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
elif page == "Prediksi Penyakit":
    st.title("Prediksi Risiko Diabetes 🩺")

    # Load data
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

    # Preprocessing data
    X = data.drop('Outcome', axis=1)

    # Load scaler (pastikan scaler disimpan saat pelatihan)
    @st.cache_resource
    def load_scaler():
        return joblib.load("scaler.pkl")  # Pastikan file scaler.pkl tersedia

    scaler = load_scaler()
    X_scaled = scaler.transform(X)

    # Load model
    @st.cache_resource
    def load_model():
        return joblib.load("svm_model .pkl")  # Pastikan path model benar

    model = load_model()

    # Prediction
    input_scaled = scaler.transform(input_df)
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

    # Tambahkan elemen interaktif: Bar Chart untuk distribusi data
    st.subheader('Distribusi Data Pengguna Dibandingkan Data Latihan')
    chart_data = pd.DataFrame({
        "Pengguna": input_df.iloc[0],
        "Rata-rata Data Latihan": data.mean()
    }).T

    st.bar_chart(chart_data)
