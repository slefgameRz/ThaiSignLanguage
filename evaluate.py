import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import logging
import random

# -----------------------------------------------------------
# 0. ตั้งค่าการทำงานและ Logging
# -----------------------------------------------------------
# กำหนด seed เพื่อให้ผลลัพธ์ใกล้เคียงกัน (แม้บาง ops บน GPU จะ nondeterministic)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ตรวจสอบ GPU และตั้ง memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info("พบ GPU: %s", gpus)
    except RuntimeError as e:
        logging.error("เกิดข้อผิดพลาดในการตั้งค่า GPU: %s", e)
else:
    logging.info("ไม่พบ GPU ใช้งาน CPU แทน")

# -----------------------------------------------------------
# 1. โหลดข้อมูล Preprocessed
# -----------------------------------------------------------
data = np.load("combined_keypoints.npz")
X_full = data["X_train"]
y_full = data["y_train"]

logging.info("โหลดข้อมูลเรียบร้อย: X_full shape: %s, y_full shape: %s", X_full.shape, y_full.shape)

# หากข้อมูลเป็น 2 มิติ (num_samples, feature_dim) -> reshape เป็น 3 มิติ (num_samples, 1, feature_dim)
if len(X_full.shape) == 2:
    X_full = X_full.reshape(X_full.shape[0], 1, X_full.shape[1])
    logging.info("Reshape X_full -> shape: %s", X_full.shape)

# -----------------------------------------------------------
# (Optional) Normalize ข้อมูล Keypoints
# -----------------------------------------------------------
mean_val = X_full.mean(axis=(0, 1))
std_val = X_full.std(axis=(0, 1)) + 1e-8
X_full = (X_full - mean_val) / std_val
logging.info("Normalized X_full ด้วย mean/std shape: %s, %s", mean_val.shape, std_val.shape)

# -----------------------------------------------------------
# 2. กำหนด Few-shot Parameters
# -----------------------------------------------------------
n_way = 5       # จำนวนคลาสในแต่ละ episode
k_shot = 5      # จำนวนตัวอย่างใน support set ต่อคลาส
q_query = 5     # จำนวนตัวอย่างใน query set ต่อคลาส
num_episodes = 10000  # จำนวน episode สำหรับการเทรน

# -----------------------------------------------------------
# (Optional) แบ่งคลาสเป็น train_classes กับ val_classes
# -----------------------------------------------------------
classes = np.unique(y_full)
np.random.shuffle(classes)
split_ratio = 0.8  # 80% ของคลาสเป็น train, 20% เป็น val
split_index = int(len(classes) * split_ratio)
train_classes = classes[:split_index]
val_classes = classes[split_index:]

train_mask = np.isin(y_full, train_classes)
val_mask = np.isin(y_full, val_classes)

X_train, y_train = X_full[train_mask], y_full[train_mask]
X_val, y_val = X_full[val_mask], y_full[val_mask]

logging.info("Train split: X=%s, y=%s | Val split: X=%s, y=%s",
             X_train.shape, y_train.shape, X_val.shape, y_val.shape)

# -----------------------------------------------------------
# ฟังก์ชัน Data Augmentation สำหรับ Keypoints
# -----------------------------------------------------------
def augment_keypoints(sequence, 
                      shift_range=0.05, 
                      scale_range=0.05, 
                      flip_prob=0.3, 
                      rotate_prob=0.3, 
                      rotate_angle=10.0):
    """
    รับค่า sequence (shape: (timesteps, features)) 
    ทำการ Augment ด้วย:
    1. Random Shift
    2. Random Scale
    3. Random Horizontal Flip (ตามค่า flip_prob)
    4. Random Rotation (สมมติหมุนเฉพาะแกน x,y รอบจุดศูนย์กลาง (0.5,0.5))
    """
    # 1) Random Shift
    shift_val = np.random.uniform(-shift_range, shift_range, size=(sequence.shape[-1],))
    sequence = sequence + shift_val

    # 2) Random Scale
    scale_val = 1.0 + np.random.uniform(-scale_range, scale_range)
    sequence = sequence * scale_val

    # 3) Random Horizontal Flip
    if np.random.rand() < flip_prob:
        # สมมติว่า feature 0 คือ x
        sequence[..., 0] = 1.0 - sequence[..., 0]

    # 4) Random Rotation (2D) สมมติหมุนเฉพาะแกน x,y
    if np.random.rand() < rotate_prob:
        angle_rad = np.radians(np.random.uniform(-rotate_angle, rotate_angle))
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        # สมมติ feature 0 = x, feature 1 = y
        x = sequence[..., 0]
        y = sequence[..., 1]
        x_center = 0.5  # หมุนรอบจุดกลาง
        y_center = 0.5
        # translate -> rotate -> translate back
        x_shifted = x - x_center
        y_shifted = y - y_center
        x_rotated = x_shifted * cos_a - y_shifted * sin_a
        y_rotated = x_shifted * sin_a + y_shifted * cos_a
        x_new = x_rotated + x_center
        y_new = y_rotated + y_center
        sequence[..., 0] = x_new
        sequence[..., 1] = y_new

    return sequence

# -----------------------------------------------------------
# 3. ฟังก์ชันสร้าง Episode สำหรับ Few-shot Learning
# -----------------------------------------------------------
def create_episode(X, y, n_way, k_shot, q_query, class_list=None, augment=False):
    """
    สุ่มเลือกคลาส (จาก class_list ถ้ามี) และตัวอย่างในคลาสเพื่อสร้าง support/query
    พร้อมตัวเลือก augment=True เพื่อทำ Data Augmentation
    """
    if class_list is None:
        class_list = np.unique(y)
    selected_classes = np.random.choice(class_list, size=n_way, replace=False)
    
    support_samples, query_samples = [], []
    support_labels, query_labels = [], []

    for new_label, cls in enumerate(selected_classes):
        indices = np.where(y == cls)[0]
        if len(indices) < (k_shot + q_query):
            chosen = np.random.choice(indices, k_shot + q_query, replace=True)
        else:
            chosen = np.random.choice(indices, k_shot + q_query, replace=False)
        
        s_part = X[chosen[:k_shot]]
        q_part = X[chosen[k_shot:]]

        if augment:
            for i in range(k_shot):
                s_part[i] = augment_keypoints(s_part[i])
            for i in range(q_query):
                q_part[i] = augment_keypoints(q_part[i])

        support_samples.append(s_part)
        query_samples.append(q_part)
        support_labels.append(np.full((k_shot,), new_label))
        query_labels.append(np.full((q_query,), new_label))

    support_samples = np.concatenate(support_samples, axis=0)
    query_samples = np.concatenate(query_samples, axis=0)
    support_labels = np.concatenate(support_labels, axis=0)
    query_labels = np.concatenate(query_labels, axis=0)

    return support_samples, support_labels, query_samples, query_labels

def episode_generator(X, y, n_way, k_shot, q_query, class_list=None, augment=False):
    """
    Generator ที่สุ่มสร้าง episode อย่างไม่สิ้นสุด
    พร้อมตัวเลือก augment=True เพื่อทำ Data Augmentation
    """
    while True:
        yield create_episode(X, y, n_way, k_shot, q_query, class_list, augment)

# สร้าง Dataset สำหรับ train (augment=True) และ val (augment=False)
train_dataset = tf.data.Dataset.from_generator(
    lambda: episode_generator(X_train, y_train, n_way, k_shot, q_query, train_classes, augment=True),
    output_signature=(
        tf.TensorSpec(shape=(n_way * k_shot, X_train.shape[1], X_train.shape[2]), dtype=tf.float32),
        tf.TensorSpec(shape=(n_way * k_shot,), dtype=tf.int64),
        tf.TensorSpec(shape=(n_way * q_query, X_train.shape[1], X_train.shape[2]), dtype=tf.float32),
        tf.TensorSpec(shape=(n_way * q_query,), dtype=tf.int64)
    )
).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: episode_generator(X_val, y_val, n_way, k_shot, q_query, val_classes, augment=False),
    output_signature=(
        tf.TensorSpec(shape=(n_way * k_shot, X_val.shape[1], X_val.shape[2]), dtype=tf.float32),
        tf.TensorSpec(shape=(n_way * k_shot,), dtype=tf.int64),
        tf.TensorSpec(shape=(n_way * q_query, X_val.shape[1], X_val.shape[2]), dtype=tf.float32),
        tf.TensorSpec(shape=(n_way * q_query,), dtype=tf.int64)
    )
).prefetch(tf.data.AUTOTUNE)

# -----------------------------------------------------------
# 4. สร้าง Encoder Model สำหรับ Few-shot Learning
# -----------------------------------------------------------
def create_encoder(timesteps, num_coordinates, embedding_dim=128):
    """
    โมเดล LSTM แบบ bidirectional ที่ซับซ้อนขึ้น
    """
    inputs = tf.keras.Input(shape=(timesteps, num_coordinates))
    
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(embedding_dim, activation='relu')(x)
    
    return tf.keras.Model(inputs, outputs)

encoder = create_encoder(X_train.shape[1], X_train.shape[2], embedding_dim=128)
encoder.summary()

# -----------------------------------------------------------
# 5. กำหนด Optimizer และ Loss function
# -----------------------------------------------------------
initial_learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

def compute_prototypes(embeddings, labels, n_way):
    prototypes = []
    for c in range(n_way):
        class_embeddings = tf.boolean_mask(embeddings, tf.equal(labels, c))
        prototypes.append(tf.reduce_mean(class_embeddings, axis=0))
    return tf.stack(prototypes)

def prototypical_loss(support_embeddings, support_labels, query_embeddings, query_labels, n_way):
    prototypes = compute_prototypes(support_embeddings, support_labels, n_way)
    distances = tf.reduce_sum((tf.expand_dims(query_embeddings, 1) - tf.expand_dims(prototypes, 0)) ** 2, axis=2)
    logits = -distances
    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(query_labels, logits, from_logits=True))
    predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, query_labels), tf.float32))
    return loss, accuracy

# ฟังก์ชัน Gradient Clipping
def clip_gradients(grads, clip_value=1.0):
    return [tf.clip_by_value(g, -clip_value, clip_value) if g is not None else g for g in grads]

# -----------------------------------------------------------
# 6. Train/Val Step
# -----------------------------------------------------------
@tf.function
def train_step(support_samples, support_labels, query_samples, query_labels):
    with tf.GradientTape() as tape:
        support_embeddings = encoder(support_samples, training=True)
        query_embeddings = encoder(query_samples, training=True)
        loss, accuracy = prototypical_loss(support_embeddings, support_labels, query_embeddings, query_labels, n_way)
    gradients = tape.gradient(loss, encoder.trainable_variables)
    gradients = clip_gradients(gradients, clip_value=1.0)
    optimizer.apply_gradients(zip(gradients, encoder.trainable_variables))
    return loss, accuracy

@tf.function
def val_step(support_samples, support_labels, query_samples, query_labels):
    support_embeddings = encoder(support_samples, training=False)
    query_embeddings = encoder(query_samples, training=False)
    loss, accuracy = prototypical_loss(support_embeddings, support_labels, query_embeddings, query_labels, n_way)
    return loss, accuracy

# -----------------------------------------------------------
# 7. Callbacks: Learning Rate Scheduler, Early Stopping
# -----------------------------------------------------------
class LRScheduler:
    def __init__(self, optimizer, factor=0.5, patience=5, min_lr=1e-6):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_val_loss = np.inf
        self.wait = 0

    def update(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = self.optimizer.learning_rate.numpy()
                new_lr = max(old_lr * self.factor, self.min_lr)
                self.optimizer.learning_rate.assign(new_lr)
                logging.info("Reduce LR from %.6f to %.6f", old_lr, new_lr)
                self.wait = 0

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_val_loss = np.inf
        self.wait = 0
        self.should_stop = False

    def update(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True

lr_scheduler = LRScheduler(optimizer, factor=0.5, patience=5, min_lr=1e-6)
early_stopper = EarlyStopping(patience=15)

# -----------------------------------------------------------
# 8. Training & Validation Loop
# -----------------------------------------------------------
episode_losses = []
episode_accuracies = []
val_losses = []
val_accuracies = []

val_interval = 100  # ทำ validation ทุก ๆ 100 episode

train_iter = iter(train_dataset)
val_iter = iter(val_dataset)

for episode in range(num_episodes):
    support_samples, support_labels, query_samples, query_labels = next(train_iter)
    loss, accuracy = train_step(support_samples, support_labels, query_samples, query_labels)
    episode_losses.append(loss.numpy())
    episode_accuracies.append(accuracy.numpy())

    if (episode + 1) % val_interval == 0:
        vsupport_samples, vsupport_labels, vquery_samples, vquery_labels = next(val_iter)
        vloss, vaccuracy = val_step(vsupport_samples, vsupport_labels, vquery_samples, vquery_labels)
        val_losses.append(vloss.numpy())
        val_accuracies.append(vaccuracy.numpy())
        
        logging.info(
            "Episode %d/%d | LR: %.6f | Train Loss: %.4f, Train Acc: %.4f | Val Loss: %.4f, Val Acc: %.4f",
            episode+1, num_episodes, optimizer.learning_rate.numpy(), loss.numpy(), accuracy.numpy(), vloss.numpy(), vaccuracy.numpy()
        )

        lr_scheduler.update(vloss.numpy())
        early_stopper.update(vloss.numpy())

        if early_stopper.should_stop:
            logging.info("Early stopping triggered at episode %d", episode+1)
            break
    else:
        if (episode + 1) % 100 == 0:
            logging.info(
                "Episode %d/%d | LR: %.6f | Train Loss: %.4f, Train Acc: %.4f",
                episode+1, num_episodes, optimizer.learning_rate.numpy(), loss.numpy(), accuracy.numpy()
            )

# -----------------------------------------------------------
# 9. Visualization
# -----------------------------------------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(episode_losses, label='Train Loss')
plt.title('Train Loss per Episode')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(episode_accuracies, label='Train Accuracy', color='orange')
plt.title('Train Accuracy Over Episodes')
plt.xlabel('Episode')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

if val_losses and val_accuracies:
    plt.figure(figsize=(12, 5))
    val_x = list(range(val_interval, (len(val_losses)*val_interval)+1, val_interval))
    
    plt.subplot(1, 2, 1)
    plt.plot(val_x, val_losses, label='Val Loss')
    plt.title('Validation Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_x, val_accuracies, label='Val Accuracy', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------
# 10. Save Model
# -----------------------------------------------------------
encoder.save("fewshot_encoder_model.h5")
logging.info("Few-shot encoder model saved as 'fewshot_encoder_model.h5'")
