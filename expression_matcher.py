import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import filedialog

# Define labels once globally
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Emotion prediction function
def predict_emotion_from_image(image, model):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, (48, 48))            # Resize image
    normalized = resized / 255.0                    # Normalize pixel values
    reshaped = np.reshape(normalized, (1, 48, 48, 1))  # Reshape to match model input
    prediction = model.predict(reshaped)            # Predict with model
    return np.argmax(prediction)

# Compare uploaded image with live webcam feed
def compare_uploaded_with_webcam():
    model = load_model("model/emotion_recognition_model.h5")

    # File dialog to upload an image
    upload_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not upload_path:
        print("❌ No image selected.")
        return

    uploaded_image = cv2.imread(upload_path)
    uploaded_emotion_idx = predict_emotion_from_image(uploaded_image, model)
    if uploaded_emotion_idx is None:
        print("❌ No face detected in uploaded image.")
        return

    uploaded_emotion = emotion_labels[uploaded_emotion_idx]
    print(f"🖼️ Uploaded emotion: {uploaded_emotion}")

    # Webcam feed
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