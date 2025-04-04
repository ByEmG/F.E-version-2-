import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def run_unsupervised():
    dataset_path = "../dataset/train"
    images = []

    # Load and flatten grayscale images
    for emotion_folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, emotion_folder)
        if os.path.isdir(folder_path):
            for img_file in os.listdir(folder_path)[:50]:
                img_path = os.path.join(folder_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (48, 48))
                images.append(img.flatten())

    images = np.array(images)
    if len(images) == 0:
        print("❌ No images loaded.")
        return

    # PCA reduction
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(images)

    # KMeans clustering
    kmeans = KMeans(n_clusters=7, random_state=42)
    clusters = kmeans.fit_predict(reduced_data)

    # Manually assign emotion labels per cluster index
    cluster_emotions = {
        0: "Angry",
        1: "Happy",
        2: "Sad",
        3: "Neutral",
        4: "Fear",
        5: "Disgust",
        6: "Surprise"
    }

    # Create scatter plot with emotion legends
    plt.figure(figsize=(10, 6))
    for cluster_id in range(7):
        cluster_points = reduced_data[clusters == cluster_id]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=cluster_emotions[cluster_id], alpha=0.7)

    plt.title("Facial Emotion Clustering (KMeans + PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Clustered Emotions")
    plt.grid(True)
    plt.show()