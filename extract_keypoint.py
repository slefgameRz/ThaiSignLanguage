import os
import cv2
import mediapipe as mp
import numpy as np
import sys
import contextlib
import logging

# ตั้งค่า logging เพื่อแสดงข้อมูลการทำงานและ error
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ลดการแสดงผล log จาก TensorFlow เพื่อความสะอาดของ output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

@contextlib.contextmanager
def suppress_output():
    """
    Context manager สำหรับระงับการแสดงผล stdout และ stderr
    ใช้เพื่อซ่อน warning ที่เกิดจาก Mediapipe ในช่วงประมวลผล
    """
    try:
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = devnull
            sys.stderr = devnull
            yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    """
    ดึง keypoints จากผลลัพธ์ของ Mediapipe Holistic:
      - pose: 33 จุด (x, y, z, visibility)
      - left hand: 21 จุด (x, y, z)
      - right hand: 21 จุด (x, y, z)
      - face: 468 จุด (x, y, z)
    คืนค่าเป็น numpy array ที่ถูก flatten
    """
    try:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten() \
               if results.pose_landmarks else np.zeros(33 * 4)
    except Exception as e:
        logging.error(f"Error extracting pose landmarks: {e}")
        pose = np.zeros(33 * 4)
    try:
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() \
                    if results.left_hand_landmarks else np.zeros(21 * 3)
    except Exception as e:
        logging.error(f"Error extracting left hand landmarks: {e}")
        left_hand = np.zeros(21 * 3)
    try:
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() \
                     if results.right_hand_landmarks else np.zeros(21 * 3)
    except Exception as e:
        logging.error(f"Error extracting right hand landmarks: {e}")
        right_hand = np.zeros(21 * 3)
    try:
        face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten() \
               if results.face_landmarks else np.zeros(468 * 3)
    except Exception as e:
        logging.error(f"Error extracting face landmarks: {e}")
        face = np.zeros(468 * 3)
    return np.concatenate([pose, left_hand, right_hand, face])

def read_image_with_utf8(path):
    """
    อ่านภาพที่มีชื่อไฟล์เป็นภาษาไทยหรือมีตัวอักษรพิเศษ
    คืนค่าเป็น image array หรือ None หากเกิดข้อผิดพลาด
    """
    try:
        with open(path, "rb") as file:
            data = np.frombuffer(file.read(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            logging.warning(f"cv2.imdecode ล้มเหลวสำหรับไฟล์: {path}")
        return image
    except Exception as e:
        logging.error(f"Error reading image {path}: {e}")
        return None

def is_image_file(filename):
    """
    ตรวจสอบว่าไฟล์เป็นไฟล์รูปภาพหรือไม่ โดยดูจากนามสกุลไฟล์
    """
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    ext = os.path.splitext(filename)[1].lower()
    return ext in valid_extensions

def extract_keypoints_from_dataset(dataset_path, save_path):
    """
    - ตรวจสอบและวนลูปอ่านภาพจากแต่ละโฟลเดอร์ (แต่ละพยัญชนะ)
    - สำหรับแต่ละภาพ ทำการประมวลผลด้วย Mediapipe Holistic
    - สร้าง label โดยใช้ชื่อโฟลเดอร์และชื่อไฟล์ (ตัดนามสกุล)
    - บันทึกข้อมูล keypoints และ label เป็นไฟล์ .npz
    """
    if not os.path.exists(dataset_path):
        logging.error(f"Dataset path ไม่ถูกต้อง: {dataset_path}")
        return

    keypoints_all = []
    labels = []

    actions = sorted(os.listdir(dataset_path))
    if not actions:
        logging.error("ไม่พบโฟลเดอร์ใน dataset")
        return

    # สร้าง instance ของ Mediapipe Holistic เพียงครั้งเดียว
    with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
        for action in actions:
            action_path = os.path.join(dataset_path, action)
            if not os.path.isdir(action_path):
                logging.info(f"ข้าม item ที่ไม่ใช่โฟลเดอร์: {action_path}")
                continue

            images = sorted(os.listdir(action_path))
            if not images:
                logging.warning(f"ไม่พบไฟล์ภาพในโฟลเดอร์: {action_path}")
                continue

            for image_filename in images:
                if not is_image_file(image_filename):
                    logging.info(f"ข้ามไฟล์ที่ไม่ใช่ภาพ: {image_filename}")
                    continue

                image_path = os.path.join(action_path, image_filename)
                image = read_image_with_utf8(image_path)
                if image is None:
                    logging.warning(f"ไม่สามารถโหลดภาพ: {image_path} จึงข้ามไป")
                    continue

                try:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    logging.error(f"Error converting image color for {image_path}: {e}")
                    continue

                image_rgb.flags.writeable = False

                # ระงับ warning จาก Mediapipe ในช่วงประมวลผล
                with suppress_output():
                    try:
                        results = holistic.process(image_rgb)
                    except Exception as e:
                        logging.error(f"Error processing image with Mediapipe for {image_path}: {e}")
                        continue

                keypoint = extract_keypoints(results)
                # สร้าง label จากชื่อโฟลเดอร์และชื่อไฟล์ (ตัดนามสกุล)
                base_name = os.path.splitext(image_filename)[0]
                label = f"{action}_{base_name}"
                keypoints_all.append(keypoint)
                labels.append(label)
                logging.info(f"ประมวลผลภาพสำเร็จ: {image_path} with label: {label}")

    # ตรวจสอบว่ามีข้อมูล keypoints ที่ถูกดึงออกมาหรือไม่
    if not keypoints_all:
        logging.error("ไม่พบ keypoints จาก dataset")
        return

    keypoints_all = np.array(keypoints_all)
    labels = np.array(labels)

    try:
        np.savez(save_path, X_train=keypoints_all, y_train=labels)
        logging.info(f"การดึง keypoints เสร็จสมบูรณ์ ข้อมูลถูกบันทึกใน '{save_path}'")
    except Exception as e:
        logging.error(f"Error saving preprocessed data to {save_path}: {e}")

if __name__ == "__main__":
    DATASET_PATH = "D:\\Thai_Sign_language__AI\\dataset"
    SAVE_PATH = "preprocessed_data.npz"
    extract_keypoints_from_dataset(DATASET_PATH, SAVE_PATH)
