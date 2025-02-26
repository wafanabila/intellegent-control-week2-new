import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# === 1. Baca dataset colors.csv dan latih model KNN ===
def train_knn_model():
    # Baca dataset
    df = pd.read_csv('colors.csv')

    # Hapus spasi tersembunyi pada nama kolom
    df.columns = df.columns.str.strip()

    # Pastikan dataset memiliki kolom yang benar
    expected_columns = ['color','color_name','hex','R','G','B']
    if not all(col in df.columns for col in expected_columns):
        raise ValueError(f"Dataset harus memiliki kolom {expected_columns}, tetapi ditemukan {df.columns}")

    # Ambil fitur (RGB) dan label (Color_Name)
    X = df[['R', 'G', 'B']].values
    y = df['color_name'].values

    # Normalisasi fitur RGB
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Latih model KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_scaled, y)

    # Simpan model dan scaler
    joblib.dump(knn, 'knn_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("✅ Model KNN dan scaler berhasil disimpan!")

# === 2. Fungsi untuk mendeteksi warna menggunakan webcam ===
def detect_colors():
    # Muat model KNN dan scaler
    knn = joblib.load('knn_model.pkl')
    scaler = joblib.load('scaler.pkl')

    # Baca dataset untuk validasi akurasi
    df = pd.read_csv('colors.csv')
    df.columns = df.columns.str.strip()

    # Simpan warna dalam dictionary untuk perbandingan
    color_dict = {tuple(row[['R', 'G', 'B']]): row['color_name'] for _, row in df.iterrows()}

    # Variabel untuk menghitung akurasi
    total_predictions = 0
    correct_predictions = 0

    # Inisialisasi kamera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Ambil ukuran frame
        height, width, _ = frame.shape

        # Ambil warna pixel tengah
        pixel_center = frame[height // 2, width // 2].reshape(1, -1)  # Ambil RGB tengah frame
        pixel_center_scaled = scaler.transform(pixel_center)  # Normalisasi
        color_pred = knn.predict(pixel_center_scaled)[0]  # Prediksi warna

        # Periksa apakah prediksi cocok dengan dataset
        pixel_rgb = tuple(pixel_center[0])
        closest_color = min(color_dict.keys(), key=lambda c: np.linalg.norm(np.array(c) - np.array(pixel_rgb)))
        actual_color = color_dict[closest_color]

        total_predictions += 1
        if color_pred.lower() == actual_color.lower():
            correct_predictions += 1

        # Hitung akurasi
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0

        # Tampilkan bounding box di tengah layar dengan warna yang sesuai
        cv2.rectangle(frame, (width//2 - 50, height//2 - 50), (width//2 + 50, height//2 + 50), pixel_center[0].tolist(), 2)
        cv2.putText(frame, f'Color: {color_pred}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, f'Accuracy: {accuracy:.2f}%', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Tampilkan frame
        cv2.imshow('Color Detection', frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === 3. Jalankan program ===
if __name__ == "__main__":
    print("🔄 Melatih model KNN berdasarkan dataset colors.csv...")
    train_knn_model()

    print("🎥 Membuka kamera untuk deteksi warna...")
    detect_colors()
