import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --- PAGE NAVIGATION ---
st.sidebar.title("SugarGuard")
page = st.sidebar.radio("Pilih Halaman", ["Home", "About Us", "Prediksi Penyakit Diabetes"])

# --- PAGE: HOME ---
if page == "Home":
    st.title("Selamat Datang di Aplikasi Prediksi Penyakit Diabetes")
    st.image("https://img.freepik.com/premium-vector/diabetes-control-doctor-checking-blood-sugar-level-elderly-man-using-insulin-pen_442961-229.jpg?w=826", width=826)
    st.write("""
        Aplikasi ini memanfaatkan Machine Learning untuk memprediksi apakah seseorang 
        berisiko terkena diabetes atau tidak. Anda dapat menggunakan fitur-fitur interaktif
        untuk memasukkan data dan melihat hasil prediksi berdasarkan algoritma pembelajaran mesin.
        
        Silakan pilih halaman di menu sebelah kiri untuk memulai prediksi atau belajar lebih lanjut 
        tentang aplikasi ini.
    """)

# --- PAGE: ABOUT US ---
elif page == "About Us":
    st.title("Tentang Kami")
    st.write("""
        Kami adalah tim pengembang teknologi yang berfokus pada implementasi Machine Learning untuk mendukung kesehatan. 
        Aplikasi ini dibuat untuk membantu memprediksi potensi risiko diabetes dengan cepat dan mudah. 

        Fitur utama dari aplikasi ini meliputi:
        - **Input Data Pengguna**: Anda bisa memasukkan nilai-nilai kesehatan seperti kadar glukosa, tekanan darah, dll.
        - **Prediksi**: Hasil prediksi apakah Anda berisiko terkena diabetes atau tidak.
        - **Probabilitas**: Tersedia juga probabilitas prediksi dari model.
        - **Akurasi Model**: Anda bisa melihat seberapa akurat model kami dengan menggunakan data latih.
        
        Jangan ragu untuk menghubungi kami jika ada pertanyaan lebih lanjut!
    """)

# --- PAGE: PREDIKSI PENYAKIT DIABETES ---
elif page == "Prediksi Penyakit Diabetes":
    st.title("Prediksi Penyakit Diabetes")
    
    # Load data
    @st.cache_data
    def load_data():
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        data = pd.read_csv(url, names=columns)
        return data

    data = load_data()

    st.sidebar.header('Masukkan Nilai Inputan Anda')

    # Fungsi untuk mendapatkan input pengguna
    def user_input_features():
        Pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=17, value=3)
        Glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=200, value=120)
        BloodPressure = st.sidebar.number_input('BloodPressure', min_value=0, max_value=122, value=70)
        SkinThickness = st.sidebar.number_input('SkinThickness', min_value=0, max_value=99, value=20)
        Insulin = st.sidebar.number_input('Insulin', min_value=0.0, max_value=846.0, value=79.0)
        BMI = st.sidebar.number_input('BMI', min_value=0.0, max_value=67.1, value=20.0)
        DiabetesPedigreeFunction = st.sidebar.number_input('DiabetesPedigreeFunction', min_value=0.0, max_value=2.42, value=0.47)
        Age = st.sidebar.number_input('Age', min_value=21, max_value=81, value=33)

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

    # Ambil input pengguna
    input_df = user_input_features()

    # Tampilkan data input pengguna
    st.subheader('Input Data Pengguna')
    st.write(input_df)

    # Pisahkan fitur dan target
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Standarisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Bagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Inisialisasi dan latih model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Standarisasi input pengguna
    input_scaled = scaler.transform(input_df)

    # Prediksi dan probabilitas prediksi
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # Tampilkan prediksi
    st.subheader('Prediksi')
    diabetes = np.array(['Tidak Diabetes', 'Diabetes'])
    st.write(f"Hasil prediksi: **{diabetes[prediction][0]}**")

    # Tampilkan probabilitas prediksi
    st.subheader('Probabilitas Prediksi')
    st.write(f"Probabilitas untuk 'Tidak Diabetes': {prediction_proba[0][0]*100:.2f}%")
    st.write(f"Probabilitas untuk 'Diabetes': {prediction_proba[0][1]*100:.2f}%")

    # Hitung akurasi model
    accuracy = accuracy_score(y_test, model.predict(X_test))
    st.subheader('Akurasi Model')
    st.write(f'Akurasi Model: {accuracy*100:.2f}%')

    # Tambahkan elemen interaktif: Bar Chart untuk distribusi data
    st.subheader('Distribusi Data Pengguna Dibandingkan Data Latihan')
    chart_data = pd.DataFrame({
        "Pengguna": input_df.iloc[0],
        "Rata-rata Data Latihan": data.mean()
    }).T

    st.bar_chart(chart_data)
