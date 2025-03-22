import os
import numpy as np

# ======================================================================
# 1. กำหนด Base Directory สำหรับ Training set
# ======================================================================
BASE_DIR = r"D:\Thai_Sign_language__AI\ภาษามือ"

# ตรวจสอบว่ามีโฟลเดอร์ Training set นี้อยู่หรือไม่
if not os.path.exists(BASE_DIR):
    raise FileNotFoundError(f"Base directory not found: {BASE_DIR}")

# ======================================================================
# 2. สร้างรายการโฟลเดอร์ของตัวอักษร (letters)
# ======================================================================
letter_folders = sorted([d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d)) and d != "Output"])
if not letter_folders:
    raise FileNotFoundError("No letter folders found in the training set.")

# ======================================================================
# 3. Loop ผ่านแต่ละโฟลเดอร์ตัวอักษรและรวบรวมข้อมูล keypoints
# ======================================================================

X_data = []  # รายการสำหรับเก็บข้อมูล keypoints (แต่ละค่าเป็นข้อมูลของหนึ่งเฟรม)
y_data = []  # รายการสำหรับเก็บ label (โดยใช้ชื่อของโฟลเดอร์ตัวอักษร)

for letter in letter_folders:
    letter_path = os.path.join(BASE_DIR, letter)
    print(f"Processing letter: {letter}")
    # แต่ละโฟลเดอร์ตัวอักษรคาดว่าจะมีหลายโฟลเดอร์ sequence (จำนวน 30)
    sequence_folders = sorted([d for d in os.listdir(letter_path) if os.path.isdir(os.path.join(letter_path, d))])
    if not sequence_folders:
        print(f"Warning: No sequence folders found in {letter_path}. Skipping this letter.")
        continue

    for seq in sequence_folders:
        seq_path = os.path.join(letter_path, seq)
        # ค้นหาไฟล์ .npy ในแต่ละ sequence; คาดว่าแต่ละ sequence ควรมี 30 ไฟล์
        npy_files = sorted([f for f in os.listdir(seq_path) if f.lower().endswith(".npy")], key=lambda x: int(os.path.splitext(x)[0]))
        if not npy_files:
            print(f"Warning: No .npy files found in {seq_path}.")
            continue
        
        for npy_file in npy_files:
            file_path = os.path.join(seq_path, npy_file)
            try:
                keypoint_data = np.load(file_path)
                # ตรวจสอบว่าข้อมูล keypoints มีค่าเหมาะสมหรือไม่
                # (คุณอาจต้องปรับให้ตรงตาม dimension ที่คาดไว้ เช่น 63, 1662, หรืออื่น ๆ)
                if keypoint_data.ndim > 1:
                    keypoint_flat = keypoint_data.flatten()
                else:
                    keypoint_flat = keypoint_data
                X_data.append(keypoint_flat)
                y_data.append(letter)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

# ======================================================================
# 4. ตรวจสอบข้อมูลที่รวบรวมได้
# ======================================================================
if len(X_data) == 0:
    print("No data was consolidated. Please check the directory structure.")
    exit(1)
else:
    print(f"Consolidated {len(X_data)} samples.")

X_array = np.array(X_data)
y_array = np.array(y_data)
print("X_array shape:", X_array.shape)
print("y_array shape:", y_array.shape)

# ======================================================================
# 5. สร้างไฟล์ npz สำหรับเก็บข้อมูล Preprocessed
# ======================================================================
output_filename = "consolidated_data.npz"
np.savez(output_filename, X=X_array, y=y_array)
print(f"Consolidated data saved to '{output_filename}'")
