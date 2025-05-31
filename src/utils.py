import os
import cv2
import numpy as np

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
    return images

def get_emotion_folders(base_path):
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    return {emotion: os.path.join(base_path, emotion) for emotion in emotions}
