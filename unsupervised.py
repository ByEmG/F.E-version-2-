import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
def run_unsupervised():
    dataset_path = "../dataset/train"
    images = []

    for emotion_folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, emotion_folder)
        if os.path.isdir(folder_path):
            for img_file in os.listdir(folder_path)[:50]:
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # ✅ Check if image was read successfully
                if img is None:
                    print(f"❌ Could not read image: {img_path}")
                    continue  # skip this image

                img = cv2.resize(img, (48, 48))
                images.append(img.flatten())

    images = np.array(images)

    if len(images) == 0:
        print("❌ No images found to cluster.")
        return

    kmeans = KMeans(n_clusters=7, random_state=42)
    clusters = kmeans.fit_predict(images)

    plt.scatter(images[:, 0], images[:, 1], c=clusters, cmap='viridis')
    plt.title("Facial Emotion Clustering (K-Means - Unsupervised)")
    plt.show()