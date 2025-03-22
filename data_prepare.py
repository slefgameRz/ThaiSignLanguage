import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf

# โหลดข้อมูลจาก CSV
df = pd.read_csv("dataset.csv")

# แยก features (คีย์พอยท์) และ labels (คำไทย)
X = df.drop("label", axis=1).values  # shape: (num_samples, 63)
y = df["label"].values               # shape: (num_samples,)

# แบ่งข้อมูลเป็นชุดฝึกและชุดทดสอบ (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # รักษาสัดส่วนคลาสใน train/test
)

# แปลงป้ายกำกับ (label) เป็นตัวเลข
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)  # เช่น "ก" -> 0, "ข" -> 1
y_test_encoded = le.transform(y_test)

# ปรับขนาดข้อมูลให้อยู่ระหว่าง 0-1 (Normalization)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ตรวจสอบข้อมูล
print("=====================================")
print("ข้อมูลตัวอย่าง 5 แถวแรก:")
print(df.head())
print("\nสถิติข้อมูล:")
print(f"จำนวนตัวอย่างทั้งหมด: {len(df)}")
print(f"จำนวนคลาส: {len(le.classes_)}")
print(f"ตัวอย่างคลาส: {le.classes_}")
print(f"ขนาดข้อมูลฝึก: {X_train_scaled.shape}")
print(f"ขนาดข้อมูลทดสอบ: {X_test_scaled.shape}")
print("=====================================")



# ใช้พารามิเตอร์ on_bad_lines แทน error_bad_lines
df = pd.read_csv(
    "dataset.csv", 
    na_values=['U', 'error_value'], 
    on_bad_lines='skip'  # ข้ามแถวที่มีข้อมูลผิดรูปแบบ
)

# แปลงคอลัมน์เป็นตัวเลข และลบแถวที่มีค่า NaN
df = df.apply(pd.to_numeric, errors='coerce').dropna()