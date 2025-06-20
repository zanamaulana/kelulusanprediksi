import streamlit as st
import pandas as pd
import joblib

# Load the trained model and encoders
try:
    model = joblib.load('prediksi_kelulusan.pkl')
    encoders = joblib.load('encoders.pkl')
except FileNotFoundError:
    st.error("Pastikan file 'prediksi_kelulusan.pkl' dan 'encoders.pkl' berada di direktori yang sama.")
    st.stop()

st.title('Aplikasi Prediksi Kelulusan Mahasiswa')
st.write('Aplikasi ini memprediksi status kelulusan mahasiswa berdasarkan beberapa fitur.')

# Input fields for user
st.sidebar.header('Input Data Mahasiswa')

# Create a dictionary to store input values
input_data = {}

# Get unique values for dropdowns from encoders (if available)
def get_unique_values(encoder_key, default_options):
    if encoder_key in encoders:
        return list(encoders[encoder_key].classes_)
    return default_options

# Numerical inputs
input_data['UMUR'] = st.sidebar.number_input('Umur', min_value=17, max_value=60, value=20)
input_data['IPK'] = st.sidebar.number_input('IPK', min_value=0.0, max_value=4.0, value=3.0, format="%.2f")
input_data['JUMLAH MATA KULIAH'] = st.sidebar.number_input('Jumlah Mata Kuliah', min_value=1, max_value=200, value=100)
input_data['SKS'] = st.sidebar.number_input('Jumlah SKS', min_value=1, max_value=300, value=144)
input_data['TAHUN MASUK'] = st.sidebar.number_input('Tahun Masuk', min_value=2000, max_value=2025, value=2020)


# Categorical inputs using selectbox
jenis_kelamin_options = get_unique_values('JENIS KELAMIN', ['Laki-laki', 'Perempuan'])
input_data['JENIS KELAMIN'] = st.sidebar.selectbox('Jenis Kelamin', jenis_kelamin_options)

status_mahasiswa_options = get_unique_values('STATUS MAHASISWA', ['Aktif', 'Cuti', 'Non-aktif'])
input_data['STATUS MAHASISWA'] = st.sidebar.selectbox('Status Mahasiswa', status_mahasiswa_options)

status_nikah_options = get_unique_values('STATUS NIKAH', ['Belum Menikah', 'Menikah'])
input_data['STATUS NIKAH'] = st.sidebar.selectbox('Status Nikah', status_nikah_options)

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

st.subheader('Data Mahasiswa yang Dimasukkan:')
st.write(input_df)

# Preprocess the input data
processed_input_df = input_df.copy()
for col in ['JENIS KELAMIN', 'STATUS MAHASISWA', 'STATUS NIKAH']:
    if col in encoders:
        try:
            processed_input_df[col] = encoders[col].transform(processed_input_df[col])
        except ValueError as e:
            st.error(f"Error transforming column {col}: {e}. Pastikan input valid.")
            st.stop()
    else:
        st.warning(f"Encoder for '{col}' not found. Skipping transformation for this column.")

# --- START OF FIX ---
# Define the exact order of features (columns) as expected by the trained model
# This order comes from your X = df.drop(columns=["STATUS KELULUSAN"]) in the notebook
expected_features = [
    'JENIS KELAMIN', 'UMUR', 'IPK', 'JUMLAH MATA KULIAH',
    'SKS', 'STATUS MAHASISWA', 'STATUS NIKAH', 'TAHUN MASUK'
]

# Reindex the DataFrame to match the expected feature order
# This ensures that the model receives columns in the correct sequence.
# Use .reindex() and then .fillna(0) for any potential missing columns if input_df was incomplete,
# although with all st.number_input/selectbox, all should be present.
processed_input_df = processed_input_df.reindex(columns=expected_features)

# Handle potential NaN values if reindex introduced them (though unlikely with current inputs)
# For simplicity, filling with 0, but consider better imputation if applicable.
processed_input_df.fillna(0, inplace=True)
# --- END OF FIX ---


st.subheader('Data Setelah Preprocessing (untuk Model):')
st.write(processed_input_df)

# Make prediction
if st.button('Prediksi Kelulusan'):
    if model:
        prediction = model.predict(processed_input_df)
        prediction_proba = model.predict_proba(processed_input_df)

        # Decode the prediction
        status_kelulusan_encoder = encoders.get('STATUS KELULUSAN')
        if status_kelulusan_encoder:
            predicted_status = status_kelulusan_encoder.inverse_transform(prediction)[0]
            st.subheader('Hasil Prediksi:')
            st.success(f'Status Kelulusan Diprediksi: **{predicted_status}**')

            st.subheader('Probabilitas Prediksi:')
            # Display probabilities for each class
            proba_df = pd.DataFrame(prediction_proba, columns=status_kelulusan_encoder.classes_)
            st.write(proba_df)
        else:
            st.warning("Encoder untuk 'STATUS KELULUSAN' tidak ditemukan. Menampilkan hasil prediksi mentah.")
            st.write(f"Prediksi Mentah: {prediction[0]}")
    else:
        st.error("Model belum dimuat. Silakan periksa kembali file model.")
