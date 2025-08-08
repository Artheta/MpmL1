from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Muat pipeline model yang sudah dilatih
# Pipeline ini sudah termasuk pra-pemrosesan, jadi bisa menerima data mentah
try:
    model = joblib.load('engagement_model.pkl')
except FileNotFoundError:
    # Jika file model tidak ditemukan, ini akan menjadi masalah saat startup
    # Sebaiknya pastikan file ada sebelum menjalankan aplikasi
    print("Error: File model 'engagement_model.pkl' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    model = None

@app.route('/')
def home():
    """Menampilkan halaman utama dengan form input."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Menerima data dari form, membuat prediksi, dan menampilkan hasilnya."""
    if model is None:
        return render_template('index.html', prediction_text='Error: Model tidak dapat dimuat.')

    try:
        # Ambil semua data dari form HTML
        form_data = request.form.to_dict()
        
        # Buat DataFrame dari data form
        # Nama kolom harus sama persis dengan 'X' saat pelatihan
        input_df = pd.DataFrame([form_data])
        
        # Konversi tipe data yang relevan (form mengirim semua sebagai string)
        input_df['Age'] = pd.to_numeric(input_df['Age'])
        input_df['Family size'] = pd.to_numeric(input_df['Family size'])
        input_df['Pin code'] = pd.to_numeric(input_df['Pin code'])
        input_df['latitude'] = pd.to_numeric(input_df['latitude'])
        input_df['longitude'] = pd.to_numeric(input_df['longitude'])

        # Gunakan pipeline yang dimuat untuk membuat prediksi
        prediction = model.predict(input_df)[0]
        
        # Kirim hasil prediksi kembali ke halaman HTML
        return render_template('index.html', prediction_text=f'Prediksi Level Keterlibatan: {prediction}')

    except Exception as e:
        # Menangani potensi error dengan baik
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    # Jalankan aplikasi dalam mode debug untuk pengembangan
    app.run(debug=True)
