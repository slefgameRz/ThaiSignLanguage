import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs

np.object = object
# ฟังก์ชันสำหรับแปลงโมเดล TensorFlow.js เป็น Keras
def convert_tfjs_to_keras(tfjs_model_dir, output_keras_file):
    # โหลดโมเดล TensorFlow.js
    model = tfjs.converters.load_tfjs_model(tfjs_model_dir)
    
    # บันทึกเป็นโมเดล Keras (.h5)
    model.save(output_keras_file)
    
    print(f"แปลงโมเดลสำเร็จแล้ว บันทึกไว้ที่: {output_keras_file}")
    return model

# ตัวอย่างการใช้งาน
tfjs_model_dir = "C:/Users/panna/Downloads/my-pose-model"  # โฟลเดอร์ที่มีไฟล์ model.json และ weights.bin
output_keras_file = "converted_model.h5"  # ชื่อไฟล์ .h5 ที่ต้องการบันทึก

# แปลงโมเดล
keras_model = convert_tfjs_to_keras(tfjs_model_dir, output_keras_file)

# แสดงสรุปโครงสร้างโมเดล
keras_model.summary()
