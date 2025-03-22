import os
import numpy as np

def merge_npz_files(file_list, output_file):
    """
    รวมข้อมูล keypoints และ labels จากหลายไฟล์ .npz เข้าด้วยกันและบันทึกเป็นไฟล์ใหม่

    Args:
        file_list (list): รายการ path ของไฟล์ .npz ที่ต้องการรวม
        output_file (str): Path ของไฟล์ผลลัพธ์ที่รวมข้อมูลแล้ว
    """
    # Initialize empty lists for combined data
    keypoints_combined = []
    filenames_combined = []

    # Process each file
    for file in file_list:
        if not os.path.exists(file):
            print(f"Skipping {file}: File not found.")
            continue
        
        try:
            # Load .npz file
            data = np.load(file, allow_pickle=True)
            
            # Check if required keys are present
            if 'keypoints' in data and 'filenames' in data:
                print(f"Processing {file}...")
                keypoints_combined.extend(data['keypoints'])
                filenames_combined.extend(data['filenames'])
            else:
                print(f"Skipping {file}: Missing 'keypoints' or 'filenames'.")
        
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Ensure data was loaded before saving
    if keypoints_combined and filenames_combined:
        # Convert to numpy arrays with dtype=object for inhomogeneous shapes 
        keypoints_array = np.array(keypoints_combined, dtype=object)
        filenames_array = np.array(filenames_combined)
        np.savez(output_file, keypoints=keypoints_array, filenames=filenames_array)
        print(f"Combined data saved to {output_file}")
    else:
        print("No valid data to combine. Please check the input files.")

# Example usage
if __name__ == "__main__":
    # Specify the list of files to merge
    file1 = r"D:\Thai_Sign_language__AI\consolidated_data.npz"
    file2 = r"D:\Thai_Sign_language__AI\extracted_keypoints.npz"
    output_file = r"D:\Thai_Sign_language__AI\combined_data.npz"

    merge_npz_files([file1, file2], output_file)
