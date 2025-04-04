# supervised_module.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model


def run_supervised():
    model = load_model("model/emotion_recognition_model.h5")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("🚨 Webcam not accessible. Make sure no other app is using it.")
        return

    print("✅ Gerard's Real-Time Supervised Emotion Recognition Started...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # This converts to grayscale
            roi = cv2.resize(roi, (48, 48))
            roi = roi / 255.0
            roi = np.expand_dims(roi, axis=0)  # Shape becomes (1, 48, 48)
            roi = np.expand_dims(roi, axis=-1)  # Shape becomes (1, 48, 48, 1)

            prediction = model.predict(roi)
            emotion_index = np.argmax(prediction)
            emotion_text = emotion_labels[emotion_index]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        cv2.imshow("Emotion Recognition - Supervised", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()