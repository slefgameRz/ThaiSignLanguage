import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# ปิด Warning ของ TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# โหลดโมเดล Keras (.keras) ที่เทรนเสร็จแล้ว
model_path = 'final_sign_language_model.keras'  # เปลี่ยนเป็นเส้นทางไฟล์โมเดลของคุณ
model = load_model(model_path)

# โหลด StandardScaler ที่บันทึกไว้
scaler = joblib.load('scaler.pkl')

# เริ่มต้น MediaPipe Holistic
mp_holistic = mp.solutions.holistic

def extract_keypoints(image):
    """
    รับภาพ BGR จาก OpenCV และแปลงเป็น keypoints vector ความยาว 1662
    โดยใช้ MediaPipe Holistic ในการตรวจจับ landmark
    """
    # แปลงภาพเป็น RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
        results = holistic.process(image_rgb)
    
    # ตรวจจับ landmark ของร่างกาย (pose)
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33 * 4)
    
    # ตรวจจับ landmark ของมือซ้าย
    if results.left_hand_landmarks:
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        left_hand = np.zeros(21 * 3)
    
    # ตรวจจับ landmark ของมือขวา
    if results.right_hand_landmarks:
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        right_hand = np.zeros(21 * 3)
    
    # ตรวจจับ landmark ของใบหน้า
    if results.face_landmarks:
        face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(468 * 3)
    
    # รวมข้อมูลทั้งหมดให้เป็น vector ขนาด 1662
    keypoints = np.concatenate([pose, left_hand, right_hand, face])
    return keypoints

def preprocess_keypoints(keypoints):
    """
    แปลง keypoints ให้อยู่ในรูปแบบที่โมเดลต้องการ (1, 1662)
    โดยที่ข้อมูลถูก normalize ด้วย StandardScaler ที่ได้บันทึกไว้
    """
    # เปลี่ยน shape เป็น (1, 1662)
    keypoints = keypoints.reshape(1, -1)
    # ทำ scaling โดยใช้ scaler ที่โหลดมา
    keypoints = scaler.transform(keypoints)
    return keypoints

# เปิดกล้อง
cap = cv2.VideoCapture(0)

CONFIDENCE_THRESHOLD = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ดึง keypoints จากภาพ
    keypoints = extract_keypoints(frame)
    # แปลงรูปแบบ keypoints ให้เป็น (1, 1662) พร้อมทำ scaling
    input_data = preprocess_keypoints(keypoints)
    
    # ทำการทำนายจากโมเดล
    predictions = model.predict(input_data)
    predicted_class = np.argmax(predictions)
    confidence = predictions[0][predicted_class]
    
    if confidence >= CONFIDENCE_THRESHOLD:
        label = f"Predicted: {predicted_class} ({confidence * 100:.2f}%)"
    else:
        label = "Prediction: Uncertain"
    
    # แสดงผลลัพธ์บนภาพ
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Sign Language Prediction", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
