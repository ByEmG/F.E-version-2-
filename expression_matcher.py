import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import filedialog

def predict_emotion_from_image(image, model):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    roi = image[y:y+h, x:x+w]  # 🟢 Use original color image (not gray)
    roi = cv2.resize(roi, (48, 48))          # Resize to match model input
    roi = roi / 255.0                        # Normalize
    roi = np.expand_dims(roi, axis=0)       # Add batch dimension

    prediction = model.predict(roi)
    return np.argmax(prediction)

def compare_uploaded_with_webcam():
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    model = load_model("model/emotion_recognition_model.h5")

    # File dialog to upload image
    upload_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not upload_path:
        print("No image selected.")
        return

    uploaded_image = cv2.imread(upload_path)
    uploaded_emotion_idx = predict_emotion_from_image(uploaded_image, model)
    if uploaded_emotion_idx is None:
        print("❌ No face detected in uploaded image.")
        return

    uploaded_emotion = emotion_labels[uploaded_emotion_idx]
    print(f"🖼️ Uploaded emotion: {uploaded_emotion}")

    # Webcam comparison
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_emotion_idx = predict_emotion_from_image(frame, model)
        if current_emotion_idx is not None:
            current_emotion = emotion_labels[current_emotion_idx]
            match = (current_emotion == uploaded_emotion)

            status = "MATCH" if match else "NO MATCH"
            color = (0, 255, 0) if match else (0, 0, 255)

            cv2.putText(frame, f"YOU: {current_emotion}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"IMAGE: {uploaded_emotion}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, status, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Compare Expression with Uploaded Image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()