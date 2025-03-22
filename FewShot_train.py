import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ======================================================================
# 1. โหลดข้อมูล Preprocessed จากไฟล์ npz
# ======================================================================
# สมมุติว่าไฟล์ npz นี้ถูกสร้างไว้แล้วและเก็บข้อมูลในรูปแบบ:
# X_train: (num_samples, timesteps, num_coordinates)
# y_train: labels เป็น integer
data = np.load("preprocessed_data.npz")
X_full = data["X_train"]  # เราจะใช้ข้อมูล training ทั้งหมดสำหรับ few-shot learning
y_full = data["y_train"]

print("ข้อมูล Preprocessed ที่โหลด:")
print("X_train shape:", X_full.shape)
print("y_train shape:", y_full.shape)

# ======================================================================
# 2. กำหนด Few-shot Parameters
# ======================================================================
n_way   = 5    # จำนวนคลาสในแต่ละ episode
k_shot  = 5    # จำนวนตัวอย่างใน support set ในแต่ละคลาส
q_query = 5    # จำนวนตัวอย่างใน query set ในแต่ละคลาส
num_episodes = 1000

# ======================================================================
# 3. ฟังก์ชันสร้าง Episode สำหรับ Few-shot Learning (Episodic Sampling)
# ======================================================================
def create_episode(X, y, n_way, k_shot, q_query):
    """
    สุ่มสร้าง episode สำหรับ Few-shot:
      - เลือก n_way คลาสจาก labels (y)
      - สำหรับแต่ละคลาสสุ่มเลือก k_shot ตัวอย่างสำหรับ support set 
        และ q_query สำหรับ query set
    คืนค่า: support_samples, support_labels, query_samples, query_labels
    """
    classes = np.unique(y)
    selected_classes = np.random.choice(classes, size=n_way, replace=False)
    
    support_samples = []
    query_samples = []
    support_labels = []
    query_labels = []
    
    for new_label, cls in enumerate(selected_classes):
        idx = np.where(y == cls)[0]
        # ถ้าจำนวนตัวอย่างในคลาสนั้นไม่พอ ให้เลือกแบบ replace
        if len(idx) < (k_shot + q_query):
            chosen = np.random.choice(idx, k_shot + q_query, replace=True)
        else:
            chosen = np.random.choice(idx, k_shot + q_query, replace=False)
        support_idx = chosen[:k_shot]
        query_idx = chosen[k_shot:]
        support_samples.append(X[support_idx])
        query_samples.append(X[query_idx])
        # ในแต่ละ episode ให้ label ใหม่เป็น 0 ถึง n_way-1
        support_labels.append(np.full((k_shot,), new_label))
        query_labels.append(np.full((q_query,), new_label))
    
    support_samples = np.concatenate(support_samples, axis=0)
    query_samples = np.concatenate(query_samples, axis=0)
    support_labels = np.concatenate(support_labels, axis=0)
    query_labels = np.concatenate(query_labels, axis=0)
    
    return support_samples, support_labels, query_samples, query_labels

# ======================================================================
# 4. สร้าง Encoder Model สำหรับ Few-shot Learning (เพิ่มเลเยอร์ให้ลึกขึ้น)
# ======================================================================
# X_full คิดว่าเป็นข้อมูลที่มี shape (samples, timesteps, num_coordinates) เช่น (num_samples, 21, 3)
timesteps = X_full.shape[1]            # เช่น 21
num_coordinates = X_full.shape[2]        # คาดว่าเป็น 3

def create_encoder(timesteps, num_coordinates, embedding_dim=128):
    inputs = tf.keras.Input(shape=(timesteps, num_coordinates))
    
    # ชั้น LSTM แรก: 128 หน่วย, คืน sequence
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # ชั้น LSTM ที่สอง: 64 หน่วย, คืน sequence
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # เพิ่มเลเยอร์ LSTM ที่สาม: 32 หน่วย, คืน sequence
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # ชั้น LSTM สุดท้าย: 32 หน่วย, ไม่คืน sequence
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=False))(x)
    
    # ชั้น Dense เพื่อตัดมิติให้อยู่ใน embedding space
    x = tf.keras.layers.Dense(embedding_dim, activation='relu')(x)
    
    return tf.keras.Model(inputs, x)

encoder = create_encoder(timesteps, num_coordinates, embedding_dim=128)
encoder.summary()

# ======================================================================
# 5. กำหนด Optimizer สำหรับ Few-shot Learning
# ======================================================================
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# ======================================================================
# 6. ฟังก์ชันคำนวณ Prototypes และ Prototypical Loss
# ======================================================================
def compute_prototypes(embeddings, labels, n_way):
    prototypes = []
    for c in range(n_way):
        class_mask = (labels == c)
        class_embeddings = embeddings[class_mask]
        prototype_c = tf.reduce_mean(class_embeddings, axis=0)
        prototypes.append(prototype_c)
    return tf.stack(prototypes)

def prototypical_loss(support_embeddings, support_labels, query_embeddings, query_labels, n_way):
    prototypes = compute_prototypes(support_embeddings, support_labels, n_way)  # รูปแบบ: (n_way, embedding_dim)
    query_expanded = tf.expand_dims(query_embeddings, axis=1)   # (num_query, 1, embedding_dim)
    prototypes_expanded = tf.expand_dims(prototypes, axis=0)      # (1, n_way, embedding_dim)
    distances = tf.reduce_sum((query_expanded - prototypes_expanded)**2, axis=2)
    logits = -distances  # ใช้ค่า negative ของระยะห่างเป็น logits
    
    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(query_labels, logits, from_logits=True))
    pred = tf.argmax(logits, axis=1)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, query_labels), tf.float32))
    return loss, acc

# ======================================================================
# 7. Episodic Training Loop สำหรับ Few-shot Learning
# ======================================================================
episode_losses = []
episode_accuracies = []

for episode in range(num_episodes):
    support_samples, support_labels, query_samples, query_labels = create_episode(X_full, y_full, n_way, k_shot, q_query)
    
    support_samples = tf.convert_to_tensor(support_samples, dtype=tf.float32)
    query_samples = tf.convert_to_tensor(query_samples, dtype=tf.float32)
    support_labels = support_labels.astype(np.int32)
    query_labels = query_labels.astype(np.int32)
    
    with tf.GradientTape() as tape:
        support_embeddings = encoder(support_samples, training=True)
        query_embeddings = encoder(query_samples, training=True)
        loss, acc = prototypical_loss(support_embeddings, support_labels, query_embeddings, query_labels, n_way)
    
    gradients = tape.gradient(loss, encoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))
    
    episode_losses.append(loss.numpy())
    episode_accuracies.append(acc.numpy())
    
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}/{num_episodes}, Loss: {loss.numpy():.4f}, Accuracy: {acc.numpy():.4f}")

# ======================================================================
# 8. แสดงผลรายละเอียดหลังการเทรน (สถิติและกราฟ)
# ======================================================================
import matplotlib.pyplot as plt

episode_losses = np.array(episode_losses)
episode_accuracies = np.array(episode_accuracies)

print("\n===== Detailed Episodic Training Statistics =====")
print(f"Total Episodes: {num_episodes}")
print(f"Average Loss  : {episode_losses.mean():.4f}")
print(f"Min Loss      : {episode_losses.min():.4f}")
print(f"Max Loss      : {episode_losses.max():.4f}")
print(f"Average Acc.  : {episode_accuracies.mean():.4f}")
print(f"Min Acc.      : {episode_accuracies.min():.4f}")
print(f"Max Acc.      : {episode_accuracies.max():.4f}")
print("===================================================")

# Plotting the training curves
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(episode_losses, label='Episode Loss')
plt.title('Episode Loss Over Training')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(episode_accuracies, label='Episode Accuracy', color='orange')
plt.title('Episode Accuracy Over Training')
plt.xlabel('Episode')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# ======================================================================
# 9. เซฟโมเดล Few-shot Encoder
# ======================================================================
encoder.save("fewshot_encoder_model.h5")
print("\nFew-shot encoder model saved as 'fewshot_encoder_model.h5'")
