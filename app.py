import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load model jika sudah disimpan, atau latih ulang dari awal
@st.cache_resource
def load_model():
    # Load data
    df = pd.read_csv('Kelulusan Train.csv')

    # Drop kolom yang tidak digunakan
    if 'NAMA' in df.columns:
        df.drop(columns=['NAMA'], inplace=True)

    # Encode kolom kategorikal
    label_cols = ['JENIS KELAMIN', 'STATUS MAHASISWA', 'STATUS NIKAH', 'STATUS KELULUSAN']
    encoders = {}
    for col in label_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=['STATUS KELULUSAN'])
    y = df['STATUS KELULUSAN']

    model = RandomForestClassifier()
    model.fit(X, y)

    return model, encoders

model, encoders = load_model()

st.title("Prediksi Status Kelulusan Mahasiswa")

# Input user
jenis_kelamin = st.selectbox("Jenis Kelamin", encoders['JENIS KELAMIN'].classes_)
status_mahasiswa = st.selectbox("Status Mahasiswa", encoders['STATUS MAHASISWA'].classes_)
status_nikah = st.selectbox("Status Nikah", encoders['STATUS NIKAH'].classes_)

# Input fitur numerik
ipk = st.number_input("IPK", 0.0, 4.0, step=0.01)
sks = st.number_input("Jumlah SKS", 0, 160, step=1)
lama_studi = st.slider("Lama Studi (semester)", 0, 14, 8)

# Encode input
input_data = pd.DataFrame({
    'JENIS KELAMIN': [encoders['JENIS KELAMIN'].transform([jenis_kelamin])[0]],
    'STATUS MAHASISWA': [encoders['STATUS MAHASISWA'].transform([status_mahasiswa])[0]],
    'STATUS NIKAH': [encoders['STATUS NIKAH'].transform([status_nikah])[0]],
    'IPK': [ipk],
    'SKS': [sks],
    'LAMA STUDI': [lama_studi],
})

if st.button("Prediksi Kelulusan"):
    prediction = model.predict(input_data)[0]
    pred_label = encoders['STATUS KELULUSAN'].inverse_transform([prediction])[0]
    st.success(f"Prediksi: Mahasiswa kemungkinan **{pred_label}**")
