import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = 'dataset_modifikasi.xlsx'
data = pd.read_excel(file_path)

# Display the first few rows of the dataset
print(data.head())
# Display the first few rows of the dataset
st.write("Dataset Preview:")
st.write(data.head())

# Define selected columns and target
selected_columns = [
    "Provinsi", "Kriteria UMKM", "Jenis UMKM", "Lokasi Usaha", "Lama Usaha (Bulan)",
    "Status Kepemilikan Tempat Usaha", "Jumlah Karyawan", "Aset Usaha (IDR)",
    "Jumlah Pasti Aset Usaha (IDR)", "Liabilitas Usaha (IDR)", "Jumlah Pasti Liabilitas Usaha (IDR)",
    "Omset Bulanan (IDR)", "Jumlah Pasti Omset Bulanan (IDR)", "Laba Bersih (IDR)",
    "Jumlah Pasti Laba Bersih (IDR)", "Jumlah Pinjaman (IDR)", "Jumlah Pasti Jumlah Pinjaman (IDR)",
    "Rekening Bank"
]
X = data[selected_columns]
y = data['Skor Kredit']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'random_forest_model.pkl')

# Load the model
model_loaded = joblib.load('random_forest_model.pkl')

# Streamlit application
st.title("Credit Score Prediction")

# Create input fields for the features
provinsi = st.number_input("Provinsi", min_value=1)
kriteria_umkm = st.number_input("Kriteria UMKM", min_value=1)
jenis_umkm = st.number_input("Jenis UMKM", min_value=1)
lokasi_usaha = st.number_input("Lokasi Usaha", min_value=1)
lama_usaha = st.number_input("Lama Usaha (Bulan)", min_value=1)
status_kepemilikan_tempat_usaha = st.number_input("Status Kepemilikan Tempat Usaha", min_value=1)
jumlah_karyawan = st.number_input("Jumlah Karyawan", min_value=1)
aset_usaha = st.number_input("Aset Usaha (IDR)", min_value=1)
jumlah_pasti_aset_usaha = st.number_input("Jumlah Pasti Aset Usaha (IDR)", min_value=1)
liabilitas_usaha = st.number_input("Liabilitas Usaha (IDR)", min_value=1)
jumlah_pasti_liabilitas_usaha = st.number_input("Jumlah Pasti Liabilitas Usaha (IDR)", min_value=1)
omset_bulanan = st.number_input("Omset Bulanan (IDR)", min_value=1)
jumlah_pasti_omset_bulanan = st.number_input("Jumlah Pasti Omset Bulanan (IDR)", min_value=1)
laba_bersih = st.number_input("Laba Bersih (IDR)", min_value=1)
jumlah_pasti_laba_bersih = st.number_input("Jumlah Pasti Laba Bersih (IDR)", min_value=1)
jumlah_pinjaman = st.number_input("Jumlah Pinjaman (IDR)", min_value=1)
jumlah_pasti_jumlah_pinjaman = st.number_input("Jumlah Pasti Jumlah Pinjaman (IDR)", min_value=1)
rekening_bank = st.number_input("Rekening Bank", min_value=1)

# Create a prediction button
if st.button("Predict"):
    # Create a DataFrame from the input features
    input_features = [
        provinsi, kriteria_umkm, jenis_umkm, lokasi_usaha, lama_usaha, status_kepemilikan_tempat_usaha,
        jumlah_karyawan, aset_usaha, jumlah_pasti_aset_usaha, liabilitas_usaha, jumlah_pasti_liabilitas_usaha,
        omset_bulanan, jumlah_pasti_omset_bulanan, laba_bersih, jumlah_pasti_laba_bersih, jumlah_pinjaman,
        jumlah_pasti_jumlah_pinjaman, rekening_bank
    ]
    input_df =pd.DataFrame([input_features], columns=selected_columns)

    # Make a prediction
    prediction = model_loaded.predict(input_df)
    predicted_score = prediction[0]

    # Display the prediction
    st.write(f"Predicted Credit Score: {predicted_score}")

    # Determine the credit score category
    def get_credit_score_category(score):
        if 300 <= score <= 579:
            return "Poor"
        elif 580 <= score <= 669:
            return "Fair"
        elif 670 <= score <= 739:
            return "Good"
        elif 740 <= score <= 799:
            return "Very Good"
        elif 800 <= score <= 850:
            return "Excellent"
        else:
            return "Invalid Score"

    credit_score_category = get_credit_score_category(predicted_score)
    st.write(f"Credit Score Category: {credit_score_category}")
