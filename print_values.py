import cv2
import requests
import time
import signal
import sys

# RESTful API'den video URL'sini almak için fonksiyon
def get_video_url():
    response = requests.get('http://127.0.0.1:5001/get_video_url')  # Portu 5001 olarak değiştirin
    if response.status_code == 200:
        return response.json()['video_url']
    else:
        return None

# Video yakalama ve işleme
def process_video():
    global cap
    i = 0  # i değişkenini tanımlayın
    current_video_url = None  # Mevcut video URL'sini saklamak için değişken

    while True:
        new_video_url = get_video_url()
        if not new_video_url:
            print("Video URL alınamadı!")
            break

        if new_video_url != current_video_url:
            print(f"Video URL güncellendi: {new_video_url}")  # Video URL'sini yazdırın
            current_video_url = new_video_url
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(current_video_url)
            if not cap.isOpened():
                print("Video açılamadı!")
                break

        ret, frame = cap.read()
        if not ret:
            print("Video akışı bitti veya okunamadı.")
            time.sleep(1)  # Yeni URL kontrolü için kısa bir bekleme süresi
            continue

        # i değişkenini artırın ve yazdırın
        i += 1
        print(f"Döngü sayısı: {i}")
        print(current_video_url)
        cv2.imwrite("frame.jpg", frame)

        # 'q' tuşuna basılırsa çıkış yap
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(1)
# Sinyal yakalayıcıyı tanımlayın
def signal_handler(sig, frame):
    print('Çıkış sinyali alındı!')
    if cap is not None:
        cap.release()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    cap = None
    process_video()
