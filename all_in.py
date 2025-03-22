# ---------------------- #
#     Import Libraries    #
# ---------------------- #
import os
import time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# ---------------------- #
#   Fix Thai Font Issue   #
# ---------------------- #
thai_fonts = ['Tahoma', 'Garuda', 'Noto Sans Thai', 'DejaVu Sans']
available_fonts = [os.path.splitext(os.path.basename(f))[0] for f in fm.findSystemFonts()]
for font in thai_fonts:
    if font in available_fonts or font in matplotlib.font_manager.get_font_names():
        plt.rcParams['font.family'] = font
        break
else:
    print("⚠️ ไม่พบ Font ที่รองรับภาษาไทย! กรุณาติดตั้ง Font เช่น 'Noto Sans Thai'")

# ---------------------- #
#   Configuration         #
# ---------------------- #
RANDOM_STATE = 42
MAX_EPOCHS = 300               # เพิ่มจำนวน Epoch เพื่อให้โมเดลมีโอกาสเรียนรู้มากขึ้น
TARGET_ACCURACY = 0.97
BATCH_SIZE = 64
INITIAL_LR = 0.0005            # ค่าเริ่มต้นของ Learning Rate
MODEL_NAME_H5 = "sign_language_model.h5"
MODEL_NAME_NATIVE = "sign_language_model.keras"  # บันทึกในรูปแบบ native ของ Keras
SCALER_NAME = "scaler.pkl"
LABEL_ENCODER_NAME = "label_encoder.pkl"

# ---------------------- #
#   Custom Callback       #
# ---------------------- #
class TargetAccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') is not None and logs.get('val_accuracy') >= TARGET_ACCURACY:
            print(f"\n🎯 ถึงความแม่นยำ {TARGET_ACCURACY*100:.0f}% แล้ว! หยุดฝึก")
            self.model.stop_training = True

# ---------------------- #
#   Data Augmentation     #
# ---------------------- #
def augment_keypoints(data):
    # ตรวจสอบว่า data มี shape ที่เหมาะสม
    if data.ndim != 2:
        raise ValueError("Data for augmentation ต้องเป็น 2 มิติ (samples, features)")
    
    n_features = data.shape[1]
    if n_features < 14:
        print("⚠️ จำนวน features น้อยกว่า 14 อาจไม่สามารถใช้ flip ได้ตามที่คาดหวัง")
        flip_indices = list(range(0, n_features, 2))  # เลือกเฉพาะค่า x ของ keypoints
    else:
        flip_indices = [0,3,6,9,12]
    
    # Flip แนวนอน
    flipped_h = data.copy()
    flipped_h[:, flip_indices] *= -1

    # Flip แนวตั้ง: สมมุติ indices สำหรับ y อยู่ในตำแหน่งที่ 1,4,7,10,13
    if n_features >= 14:
        flipped_v = data.copy()
        flipped_v[:, [1,4,7,10,13]] *= -1
    else:
        flipped_v = data.copy()
        flipped_v[:, 1::2] *= -1

    # เพิ่ม noise เล็กน้อย
    noise = np.random.normal(0, 0.003, data.shape)
    
    # นอกเหนือจากการ augment เดิม อาจเพิ่มการเลื่อน (shift) เล็กน้อย
    shift = np.roll(data, shift=1, axis=1)  # ตัวอย่างการ shift ข้อมูล
    return np.concatenate([data + noise, flipped_h, flipped_v, shift])

# ---------------------- #
#   Main Function         #
# ---------------------- #
def main():
    try:
        # 1. ตรวจสอบไฟล์ข้อมูล
        dataset_path = "dataset.csv"
        if not os.path.exists(dataset_path):
            raise FileNotFoundError("❌ ไม่พบไฟล์ dataset.csv")
        
        # 2. โหลดข้อมูลด้วยการกำหนด na_values
        df = pd.read_csv(
            dataset_path,
            sep=',',
            header=0,
            na_values=['U', 'error_value', '#', '...', 'nan', ''],
            encoding='utf-8'
        )
        print("\n✅ โหลดข้อมูลสำเร็จ")
        print(f"จำนวนข้อมูลดั้งเดิม: {len(df)} แถว")
        print(f"จำนวน features: {len(df.columns)-1} (ไม่รวม label)")
        print("\n📜 ตัวอย่างข้อมูล 3 แถวแรก:")
        print(df.head(3))
        
        # 3. ทำความสะอาดข้อมูลด้วย fillna แทนการลบข้อมูล
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)
        print(f"\n🛠 ข้อมูลหลังการเติม Missing Values: {len(df)} แถว")
        
        if 'label' not in df.columns:
            raise KeyError("❌ ไม่พบคอลัมน์ 'label'")
        
        # 4. วิเคราะห์การกระจายตัวของคลาส
        print("\n📊 การกระจายตัวของคลาส:")
        class_dist = df['label'].value_counts()
        print(class_dist)
        
        plt.figure(figsize=(12,6))
        sns.countplot(
            x=df['label'],
            hue=df['label'],
            palette='viridis',
            legend=False
        )
        plt.title('การกระจายตัวของคลาสข้อมูล', fontsize=16, pad=20)
        plt.xlabel('คลาส', fontsize=14)
        plt.ylabel('จำนวนตัวอย่าง', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        # 5. เตรียมข้อมูลสำหรับโมเดล
        le = LabelEncoder()
        y = df["label"].values
        y_encoded = le.fit_transform(y)
        joblib.dump(le, LABEL_ENCODER_NAME)
        
        # 6. คำนวณ Class Weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_encoded),
            y=y_encoded
        )
        class_weights = dict(enumerate(class_weights))
        
        # 7. แบ่งข้อมูลเป็น Training และ Testing
        X = df.drop("label", axis=1).values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_encoded
        )
        
        # 8. ปรับขนาดข้อมูล (Standardization)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        joblib.dump(scaler, SCALER_NAME)
        
        # 9. เพิ่มข้อมูล Augmentation ใน Training Set
        try:
            X_train_augmented = augment_keypoints(X_train_scaled)
            X_train_final = np.concatenate([X_train_scaled, X_train_augmented])
            # replicate labels 5 ครั้ง (ข้อมูลเดิม + 4 ชุด augment)
            y_train_final = np.concatenate([y_train, y_train, y_train, y_train, y_train])
            print(f"\n🔀 Training data เพิ่มขึ้นจาก {X_train_scaled.shape[0]} เป็น {X_train_final.shape[0]} แถว หลังการ Augmentation")
        except Exception as aug_ex:
            print("⚠️ ไม่สามารถ augment ข้อมูลได้, ใช้ข้อมูล Training เดิมแทน:", str(aug_ex))
            X_train_final = X_train_scaled
            y_train_final = y_train
        
        # 10. สร้างโมเดล (ปรับสถาปัตยกรรมให้มีความลึกและ capacity สูงขึ้น)
        model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.25),
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(len(le.classes_), activation='softmax')
        ])
        
        # 11. คอมไพล์โมเดล
        optimizer = Adam(learning_rate=INITIAL_LR)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 12. กำหนด Callbacks
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=25,
            mode='max',
            restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.8,
            patience=15,
            min_lr=1e-6
        )
        target_acc_cb = TargetAccuracyCallback()
        
        # 13. ฝึกโมเดล
        print("\n🚀 เริ่มฝึกโมเดล...")
        start_time = time.time()
        history = model.fit(
            X_train_final, y_train_final,
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test_scaled, y_test),
            callbacks=[target_acc_cb, early_stop, reduce_lr],
            class_weight=class_weights,
            verbose=1
        )
        training_time = time.time() - start_time
        
        # 14. บันทึกโมเดลในรูปแบบ H5 และ native ของ Keras
        model.save(MODEL_NAME_H5)
        model.save(MODEL_NAME_NATIVE)
        print(f"\n💾 บันทึกไฟล์สำเร็จ:")
        print(f"- โมเดล H5: {MODEL_NAME_H5}")
        print(f"- โมเดล Native: {MODEL_NAME_NATIVE}")
        print(f"- Scaler: {SCALER_NAME}")
        print(f"- Label Encoder: {LABEL_ENCODER_NAME}")
        
        # 15. Visualization: แสดงกราฟความแม่นยำและ Loss
        plt.figure(figsize=(18, 8))
        # กราฟ Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='ความแม่นยำฝึก', linewidth=2, marker='o', markersize=5)
        plt.plot(history.history['val_accuracy'], label='ความแม่นยำตรวจสอบ', linewidth=2, marker='s', markersize=5)
        plt.title('พัฒนาการความแม่นยำ', fontsize=16, pad=20)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('ความแม่นยำ', fontsize=14)
        plt.ylim(0.5, 1.0)
        plt.axhline(y=TARGET_ACCURACY, color='r', linestyle='--', label='เป้าหมาย')
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # กราฟ Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Loss ฝึก', linewidth=2, marker='o', markersize=5)
        plt.plot(history.history['val_loss'], label='Loss ตรวจสอบ', linewidth=2, marker='s', markersize=5)
        plt.title('พัฒนาการ Loss', fontsize=16, pad=20)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        # 16. รายงานผลละเอียดการฝึก
        best_epoch = np.argmax(history.history['val_accuracy'])
        print("\n📜 รายงานผลการฝึก:")
        print(f"- เวลาการฝึก: {training_time//60:.0f} นาที {training_time%60:.2f} วินาที")
        print(f"- จำนวน Epoch ทั้งหมด: {len(history.history['accuracy'])}")
        print(f"- Epoch ที่ดีที่สุด: {best_epoch+1}")
        print(f"- ความแม่นยำสูงสุด (Validation): {max(history.history['val_accuracy'])*100:.2f}%")
        print(f"- Loss ต่ำสุด (Validation): {min(history.history['val_loss']):.4f}")
        
        # เก็บ learning rate สุดท้ายโดยใช้ attribute 'learning_rate'
        lr_val = model.optimizer.learning_rate
        if hasattr(lr_val, 'numpy'):
            lr_val = lr_val.numpy()
        print(f"- Learning Rate สุดท้าย: {lr_val:.6f}")
        
        # 17. ประเมินโมเดลบนชุดข้อมูลทดสอบ
        test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
        print("\n🧪 การทดสอบสุดท้าย:")
        print(f"- ความแม่นยำบนข้อมูลทดสอบ: {test_acc*100:.2f}%")
        print(f"- Loss บนข้อมูลทดสอบ: {test_loss:.4f}")
        
    except Exception as e:
        print("\n❌ เกิดข้อผิดพลาด:", str(e))
        print("\n🔧 วิธีแก้ไข:")
        print("- ตรวจสอบรูปแบบไฟล์และ encoding ให้ถูกต้อง")
        print("- ตรวจสอบคอลัมน์ 'label' มีอยู่ในข้อมูล")
        print("- ตรวจสอบข้อมูล Missing Values และการจัดการ Augmentation")

if __name__ == "__main__":
    main()
