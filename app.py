from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
import tensorflow as tf

# --- Konfigurasi ---
# Pastikan path ini benar sesuai dengan lokasi model Anda
MODEL_PATH = 'model/best_model_temp.h5' 
IMG_SIZE = (224, 224)
# Pastikan jumlah label sesuai dengan output model Anda (10 kelas)
LABELS = [f'mst_{i+1}' for i in range(10)]

# --- Inisialisasi Aplikasi Flask & Model ---
app = Flask(__name__)

# Memuat model saat aplikasi dimulai
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except IOError:
    print(f"Error: File model tidak ditemukan di '{MODEL_PATH}'. Pastikan path sudah benar.")
    model = None

# --- Fungsi-fungsi Pemrosesan Gambar ---

def segment_skin(img):
    """
    Segmentasi kulit dari gambar menggunakan thresholding YCrCb.
    Versi ini tidak menampilkan gambar (plt.show() dihapus).
    """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    skin = cv2.bitwise_and(img, img, mask=mask)
    
    # Mengisi latar belakang dengan warna putih, bukan hitam
    bg = np.full_like(img, 255)
    bg_mask = cv2.bitwise_not(mask)
    out = cv2.bitwise_and(skin, skin, mask=mask) + cv2.bitwise_and(bg, bg, mask=bg_mask)
    
    return out

def preprocess_image(file_stream):
    """
    Mempraproses gambar: membaca dari stream, mengubah ukuran, segmentasi, dan normalisasi.
    Versi ini tidak menampilkan gambar (plt.show() dihapus).
    """
    # Membaca gambar dari stream file yang diunggah
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Jika gambar gagal dibaca, kembalikan None
    if img is None:
        return None

    # Mengubah ukuran gambar
    img_resized = cv2.resize(img, IMG_SIZE)
    
    # Segmentasi kulit
    img_segmented = segment_skin(img_resized)
    
    # Konversi ke RGB (diperlukan untuk input model)
    img_rgb = cv2.cvtColor(img_segmented, cv2.COLOR_BGR2RGB)
    
    # Normalisasi gambar
    img_normalized = img_rgb.astype('float32') / 255.0
    
    # Menambahkan dimensi batch agar sesuai dengan bentuk input model
    return np.expand_dims(img_normalized, axis=0)

# --- Rute API ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Menerima gambar, menjalankan inferensi, dan mengembalikan hasil JSON."""
    if model is None:
        return jsonify({'error': 'Model tidak berhasil dimuat, periksa path model di server.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada bagian file dalam permintaan'}), 400
    
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

    try:
        # Mempraproses gambar dari stream file
        processed_image = preprocess_image(f.stream)
        
        if processed_image is None:
            return jsonify({'error': 'Format gambar tidak valid atau file rusak'}), 400

        # Menjalankan prediksi
        preds = model.predict(processed_image)[0]
        
        # Mendapatkan indeks dan nilai kepercayaan tertinggi
        idx = int(np.argmax(preds))
        confidence = float(preds[idx]) # Mengembalikan sebagai float, bukan string
        label = LABELS[idx]
        
        # Mengembalikan hasil sebagai respons JSON
        # Format ini sesuai dengan yang diharapkan oleh kode JavaScript di Canvas
        return jsonify({'skintone': label, 'confidence': confidence})

    except Exception as e:
        # Menangani error tak terduga selama pemrosesan
        print(f"An error occurred: {e}")
        return jsonify({'error': 'Terjadi kesalahan saat memproses gambar.'}), 500

# Menghapus rute '/' karena tidak diperlukan, front-end sudah terpisah.

if __name__ == '__main__':
    # Jalankan server Flask. 'debug=True' baik untuk pengembangan.
    app.run(debug=True)
