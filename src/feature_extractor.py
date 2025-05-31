import cv2
import numpy as np

class FacialFeatureExtractor:
    def __init__(self, output_size=(64, 64)):
        self.output_size = output_size
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def extract_features(self, face_img):
        color_face = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)

        faces = self.face_cascade.detectMultiScale(face_img, 1.1, 4)
        if len(faces) == 0:
            return None
            
        (x, y, w, h) = faces[0]
        face_roi = face_img[y:y+h, x:x+w]
        
        features = {
            'left_eye': self._extract_left_eye(face_img),
            'right_eye': self._extract_right_eye(face_img),
            'mouth': self._extract_mouth(face_img),
            'nose': self._extract_nose(face_img),
            'full_face': self._resize(face_img)
        }
        
        return features
    
    def _resize(self, img):
        return cv2.resize(img, self.output_size, interpolation=cv2.INTER_AREA)
    
    def _extract_left_eye(self, face_img):
        # 在48x48图像中，眼睛大约在上半部分的左侧
        eye_region = face_img[10:25, 5:20]
        return self._resize(eye_region)
    
    def _extract_right_eye(self, face_img):
        # 在48x48图像中，眼睛大约在上半部分的右侧
        eye_region = face_img[10:25, 28:43]
        return self._resize(eye_region)
    
    def _extract_mouth(self, face_img):
        # 嘴巴大约在下半部分
        mouth_region = face_img[30:45, 10:38]
        return self._resize(mouth_region)
    
    def _extract_nose(self, face_img):
        # 鼻子大约在中间部分
        nose_region = face_img[20:35, 15:33]
        return self._resize(nose_region)
