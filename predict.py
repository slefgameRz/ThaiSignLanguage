import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import joblib

# -------------------------------
# 1. โหลดข้อมูลและ Preprocessing
# -------------------------------
DATA_FILE = "combined_data.npz"  # ไฟล์ที่เก็บ keypoints และ filenames
expected_size = 1662             # ขนาดของ keypoints ที่คาดหวัง

# โหลดข้อมูลจากไฟล์ NPZ
data = np.load(DATA_FILE, allow_pickle=True)
X = data["keypoints"]   # แต่ละ element ควรมีขนาด (1662,) หรือใกล้เคียง
filenames = data["filenames"]

# แปลง filenames เป็น labels โดยใช้ส่วนแรกของชื่อไฟล์ (ก่อนเครื่องหมาย _)
labels = [fname.split('_')[0] for fname in filenames]
print("Extracted labels (first 10):", labels[:10])

# ปรับขนาด keypoints ให้มีความยาวเท่ากัน (trim หรือ pad ด้วย 0)
X_processed = []
for kp in X:
    kp = np.array(kp)
    if kp.shape[0] > expected_size:
        kp = kp[:expected_size]
    elif kp.shape[0] < expected_size:
        kp = np.pad(kp, (0, expected_size - kp.shape[0]), mode='constant')
    X_processed.append(kp)

# รวมข้อมูล keypoints ให้เป็น numpy array 2 มิติ (num_samples, expected_size)
X_processed = np.vstack(X_processed)
print("Processed X shape:", X_processed.shape)

# Normalize input data ด้วย StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)

# บันทึก scaler ลงไฟล์ scaler.pkl
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as scaler.pkl")

# เข้ารหัส labels เป็นตัวเลข แล้วแปลงเป็น one-hot encoding
encoder = LabelEncoder()
y_int = encoder.fit_transform(labels)
y = tf.keras.utils.to_categorical(y_int)
print("y shape:", y.shape)
print("Mapping:", dict(zip(encoder.classes_, encoder.transform(encoder.classes_))))

# แบ่งข้อมูลเป็น train/test (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# -------------------------------
# 2. สร้างโมเดล Deep Neural Network
# -------------------------------
model = Sequential([
    Dense(1024, activation='relu', input_shape=(expected_size,)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# -------------------------------------------
# 3. ตั้งค่า Callbacks สำหรับการเทรนที่ละเอียด
# -------------------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
checkpoint = ModelCheckpoint("best_sign_language_model.keras", monitor='val_loss', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

callbacks = [early_stop, checkpoint, reduce_lr]

# -------------------------------
# 4. ฝึกโมเดลและประเมินผล
# -------------------------------
history = model.fit(X_train, y_train, epochs=200, batch_size=16,
                    validation_split=0.1, callbacks=callbacks, verbose=1)

# ประเมินโมเดลด้วยชุด Test
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", loss)
print("Test accuracy:", acc)

# บันทึกโมเดลสุดท้าย
model.save("final_sign_language_model.keras")
print("Final model saved as final_sign_language_model.keras")

# -------------------------------
# 5. แสดงกราฟความคืบหน้าการเทรน
# -------------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss during Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy during Training")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
