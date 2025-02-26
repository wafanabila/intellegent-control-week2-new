import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Muat model SVM dan scaler
svm = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')

# Baca dataset untuk validasi akurasi 
df = pd.read_csv('colors.csv')
df.columns = df.columns.str.strip()

# Simpan warna dalam dictionary untuk perbandingan
color_dict = {tuple(row[['R', 'G', 'B']]): row['color_name'] for _, row in df.iterrows()}

# Definisi rentang warna dalam HSV (lower, upper)
color_ranges = {
    "Red": ([0, 120, 70], [10, 255, 255]),    # Merah
    "Green": ([36, 100, 100], [86, 255, 255]), # Hijau
    "Blue": ([94, 80, 2], [126, 255, 255]),    # Biru r g b 
    "Yellow": ([20, 100, 100], [30, 255, 255]) # Kuning
}

# Warna untuk bounding box (dalam BGR)
color_bgr = {
    "Red": (0, 0, 255),
    "Green": (0, 255, 0),
    "Blue": (255, 0, 0),
    "Yellow": (0, 255, 255)
}

# Variabel untuk menghitung akurasi
total_detections = 0
correct_predictions = 0

# Inisialisasi Kamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi frame ke HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, (lower, upper) in color_ranges.items():
        # Buat mask berdasarkan warna
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        # Temukan kontur objek berdasarkan mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Hanya proses objek besar
                x, y, w, h = cv2.boundingRect(contour)

                # Ambil warna di titik tengah bounding box
                center_x, center_y = x + w // 2, y + h // 2
                pixel_color = frame[center_y, center_x]  # Ambil RGB
                pixel_color = pixel_color[::-1]  # Konversi BGR ke RGB

                # Normalisasi dan prediksi warna dengan SVM
                pixel_color_scaled = scaler.transform([pixel_color])
                predicted_color = svm.predict(pixel_color_scaled)[0]

                # Temukan warna RGB terdekat dalam dataset menggunakan Euclidean Distance
                closest_color = min(color_dict.keys(), key=lambda c: np.linalg.norm(np.array(c) - np.array(pixel_color)))
                actual_color = color_dict[closest_color]

                # Periksa apakah prediksi benar
                total_detections += 1
                if predicted_color.lower() == actual_color.lower():
                    correct_predictions += 1

                # Hitung akurasi (hindari pembagian dengan nol)
                accuracy = (correct_predictions / total_detections * 100) if total_detections > 0 else 0

                # Gambar bounding box dan teks warna
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr[color_name], 2)
                cv2.putText(frame, f'{predicted_color} ({accuracy:.2f}%)', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr[color_name], 2)

    # Tampilkan frame dengan bounding box dan akurasi
    cv2.imshow('Color Detection with Accuracy', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
