import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load model dan label encoder (misalnya disimpan sebelumnya)
model = joblib.load('model_kelulusan.pkl')
encoders = joblib.load('encoders.pkl')  # Dictionary of LabelEncoders

st.title("Prediksi Kelulusan Mahasiswa")

# Form input
with st.form("form_kelulusan"):
    ipk = st.number_input('IPK', min_value=0.0, max_value=4.0, step=0.01)
    sks = st.number_input('Jumlah SKS', min_value=0, max_value=200, step=1)
    masa_studi = st.number_input('Masa Studi (semester)', min_value=1, max_value=14)
    kehadiran = st.slider('Persentase Kehadiran (%)', min_value=0, max_value=100)

    jenis_kelamin = st.selectbox('Jenis Kelamin', encoders['JENIS KELAMIN'].classes_)
    status_mahasiswa = st.selectbox('Status Mahasiswa', encoders['STATUS MAHASISWA'].classes_)
    status_nikah = st.selectbox('Status Nikah', encoders['STATUS NIKAH'].classes_)

    submit = st.form_submit_button("Prediksi")

# Ketika tombol ditekan
if submit:
    # Siapkan array input
    input_dict = {
        'IPK': ipk,
        'SKS': sks,
        'MASA STUDI': masa_studi,
        'KEHADIRAN': kehadiran,
        'JENIS KELAMIN': encoders['JENIS KELAMIN'].transform([jenis_kelamin])[0],
        'STATUS MAHASISWA': encoders['STATUS MAHASISWA'].transform([status_mahasiswa])[0],
        'STATUS NIKAH': encoders['STATUS NIKAH'].transform([status_nikah])[0],
    }

    features = np.array([[input_dict[col] for col in input_dict]])

    # Prediksi
    prediction = model.predict(features)[0]
    label = encoders['STATUS KELULUSAN'].inverse_transform([prediction])[0]

    # Tampilkan hasil
    if label.lower() in ['tidak lulus', 'terlambat']:
        st.error(f"Hasil Prediksi: {label}")
    else:
        st.success(f"Hasil Prediksi: {label}")
