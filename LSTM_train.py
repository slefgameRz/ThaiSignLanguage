import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras_tuner as kt  # ถ้าใช้ keras-tuner เวอร์ชัน 1.0 หรือสูงกว่า

# ======================================================================
# 1. โหลดข้อมูลจาก CSV และ Preprocess
# ======================================================================
csv_file = "keypoints.csv"  # ปรับ path ให้ถูกต้อง
df = pd.read_csv(csv_file)

print("ตรวจสอบค่า missing:")
print(df.isnull().sum())

# ฟังก์ชันแปลง keypoints จาก string เป็น numpy array ของ float32
def parse_keypoints(kp_str):
    try:
        arr = np.array([float(x) for x in kp_str.split(",")], dtype=np.float32)
        return arr
    except Exception as e:
        print("Error parsing keypoints:", kp_str)
        return None

# สมมุติว่าใน CSV คอลัมน์ "keypoints" เก็บข้อมูลเป็น string ที่มี 63 ค่า (21 timesteps * 3 พิกัด)
df["keypoints_arr"] = df["keypoints"].apply(parse_keypoints)
df = df[df["keypoints_arr"].notnull()]  # กรองแถวที่ parsing ไม่ได้

# ======================================================================
# 2. แปลง Label ด้วย LabelEncoder และ One-hot Encoding
# ======================================================================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df["class"].values)
num_classes = len(label_encoder.classes_)
y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes)

print(f"จำนวนคลาส: {num_classes}")
print("Label classes:", label_encoder.classes_)

# ======================================================================
# 3. เตรียมข้อมูล X และ Reshape เป็น (samples, timesteps, num_coordinates)
# ======================================================================
X = np.stack(df["keypoints_arr"].values)  # (num_samples, 63)
num_coords = 3
if X.shape[1] % num_coords != 0:
    raise ValueError("จำนวนฟีเจอร์ไม่หารด้วย 3 ลงตัว")
timesteps = X.shape[1] // num_coords
X = X.reshape(-1, timesteps, num_coords)
print(f"รูปแบบ X หลัง reshape: {X.shape}")

# ======================================================================
# 4. แบ่งข้อมูลเป็น Training/Validation
# ======================================================================
X_train, X_val, y_train, y_val = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
print(f"X_train: {X_train.shape}, X_val: {X_val.shape}")

# ======================================================================
# 5. สร้าง tf.data.Dataset พร้อม Data Augmentation
# ======================================================================
batch_size = 32

def augment_keypoints(x, y):
    """
    Data augmentation สำหรับ keypoints:
      - เพิ่ม noise (เบา ๆ)
      - random scaling (0.9 ถึง 1.1)
      - random rotation (เฉพาะแกน x,y)
    """
    # x shape: (timesteps, 3)
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.01)
    x = x + noise

    scale = tf.random.uniform([], 0.9, 1.1)
    x = x * scale

    angle = tf.random.uniform([], -0.1, 0.1)  # radians
    cos_val = tf.cos(angle)
    sin_val = tf.sin(angle)
    # หมุนเฉพาะแกน x,y
    xy = x[:, :2]
    z = x[:, 2:]
    rotation_matrix = tf.reshape(tf.stack([cos_val, -sin_val, sin_val, cos_val]), (2, 2))
    xy_rotated = tf.matmul(xy, rotation_matrix)
    x = tf.concat([xy_rotated, z], axis=1)
    
    return x, y

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.map(augment_keypoints, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(1000, seed=42).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ======================================================================
# 6. สร้าง Model Builder Function สำหรับ Keras Tuner (Hyperparameter Tuning)
# ======================================================================
def model_builder(hp):
    inputs = tf.keras.Input(shape=(timesteps, num_coords))
    # ชั้น LSTM แรก
    lstm_units_1 = hp.Int('lstm_units_1', min_value=64, max_value=256, step=64)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units_1, return_sequences=True))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    dropout_rate1 = hp.Float('dropout_rate1', min_value=0.3, max_value=0.5, step=0.05)
    x = tf.keras.layers.Dropout(dropout_rate1)(x)
    
    # ชั้น LSTM ที่สอง
    lstm_units_2 = hp.Int('lstm_units_2', min_value=32, max_value=128, step=32)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units_2, return_sequences=False))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    dropout_rate2 = hp.Float('dropout_rate2', min_value=0.2, max_value=0.5, step=0.05)
    x = tf.keras.layers.Dropout(dropout_rate2)(x)
    
    # ชั้น Dense
    dense_units = hp.Int('dense_units', min_value=32, max_value=128, step=32)
    x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    dropout_rate3 = hp.Float('dropout_rate3', min_value=0.2, max_value=0.5, step=0.05)
    x = tf.keras.layers.Dropout(dropout_rate3)(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# ======================================================================
# 7. ใช้ Keras Tuner เพื่อค้นหาพารามิเตอร์ที่ดีที่สุด
# ======================================================================
tuner = kt.RandomSearch(
    model_builder,
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=1,
    directory='kt_dir',
    project_name='hand_gesture_tuning'
)

print("เริ่มค้นหาพารามิเตอร์ที่ดีที่สุดด้วย Keras Tuner...")
tuner.search(train_ds, epochs=10, validation_data=val_ds,
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])

# แสดงผล best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\nBest Hyperparameters:")
for param in best_hps.values.keys():
    print(f"{param}: {best_hps.get(param)}")

# ======================================================================
# 8. สร้างโมเดลด้วย Hyperparameters ที่ดีที่สุด และเทรนต่อด้วย callbacks ขั้นสูง
# ======================================================================
model = tuner.hypermodel.build(best_hps)
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
    tf.keras.callbacks.ModelCheckpoint("best_hand_gesture_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1),
    tf.keras.callbacks.TensorBoard(log_dir="./logs")
]

epochs = 50
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

# ======================================================================
# 9. เซฟโมเดลสุดท้ายและแสดงกราฟ Training History
# ======================================================================
model.save("trained_hand_gesture_model_final.h5")
print("โมเดลถูกเซฟไว้ที่ 'trained_hand_gesture_model_final.h5'")

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
