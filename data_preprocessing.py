import os
import cv2
import numpy as np
import mediapipe as mp

# =============================================================================
# 1. Mediapipe Holistic Configuration
# =============================================================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic_config = {
    "static_image_mode": True,
    "model_complexity": 2,
    "refine_face_landmarks": True,
    "min_detection_confidence": 0.8,
    "min_tracking_confidence": 0.8
}

def mediapipe_detection(image, model):
    """
    Process an image with Mediapipe Holistic and return the results.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    return results

def extract_keypoints(results):
    """
    Extract keypoints from Mediapipe Holistic results as a concatenated 1D array.
    """
    pose = (np.array([[res.x, res.y, res.z, res.visibility]
                      for res in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(33 * 4))
    face = (np.array([[res.x, res.y, res.z]
                      for res in results.face_landmarks.landmark]).flatten()
            if results.face_landmarks else np.zeros(468 * 3))
    left_hand = (np.array([[res.x, res.y, res.z]
                           for res in results.left_hand_landmarks.landmark]).flatten()
                 if results.left_hand_landmarks else np.zeros(21 * 3))
    right_hand = (np.array([[res.x, res.y, res.z]
                            for res in results.right_hand_landmarks.landmark]).flatten()
                 if results.right_hand_landmarks else np.zeros(21 * 3))
    return np.concatenate([pose, face, left_hand, right_hand])

# =============================================================================
# 2. Process Training Set and Handle Mixed Shapes
# =============================================================================
def process_training_set(base_dir, output_npz):
    """
    Traverse subfolders in the training set and extract keypoints from images.
    """
    keypoints_list = []  # Store keypoints for each image
    filenames_list = []  # Store filenames corresponding to the keypoints

    with mp_holistic.Holistic(**holistic_config) as holistic:
        for subfolder in sorted(os.listdir(base_dir)):
            subfolder_path = os.path.join(base_dir, subfolder)
            images_dir = os.path.join(subfolder_path, "Images (JPEG)")

            if not os.path.exists(images_dir):
                print(f"Skipping: 'Images (JPEG)' not found in {subfolder_path}")
                continue

            print(f"Processing images in: {images_dir}")
            image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

            if not image_files:
                print(f"No image files found in {images_dir}. Skipping.")
                continue

            for file in image_files:
                image_path = os.path.join(images_dir, file)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue

                # Process image and extract keypoints
                results = mediapipe_detection(image, holistic)
                keypoints = extract_keypoints(results)

                keypoints_list.append(keypoints)  # Append 1D array to the list
                filenames_list.append(os.path.join(subfolder, file))

                print(f"Processed: {file} - Keypoints shape: {keypoints.shape}")

    # Save keypoints and filenames as object arrays
    if keypoints_list:
        keypoints_array = np.array(keypoints_list, dtype=object)  # Use dtype=object to allow mixed shapes
        filenames_array = np.array(filenames_list)
        np.savez(output_npz, keypoints=keypoints_array, filenames=filenames_array)
        print(f"Keypoints saved to: {output_npz}")
    else:
        print("No keypoints extracted. Please check your input data.")

# =============================================================================
# 3. Run the Script
# =============================================================================
if __name__ == "__main__":
    BASE_DIR = r"D:\Thai_Sign_language__AI\One-Stage-TFS Thai One-Stage Fingerspelling Dataset\One-Stage-TFS Thai One-Stage Fingerspelling Dataset\Training set"
    OUTPUT_NPZ = os.path.join(BASE_DIR, "extracted_keypoints.npz")

    process_training_set(BASE_DIR, OUTPUT_NPZ)
