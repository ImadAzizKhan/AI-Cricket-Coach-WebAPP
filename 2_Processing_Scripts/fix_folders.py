import os
import shutil

# Your exact path to the Raw Videos folder
BASE_PATH = r"D:\Personal Projects\VR Bat\cricket-pose-detection-analysis-main\CricShot10k_Dataset\CricShot10k_ Shot Dataset\Cricket_AI_Training\1_Raw_Videos"

for class_folder in os.listdir(BASE_PATH):
    class_dir = os.path.join(BASE_PATH, class_folder)
    
    # Make sure we are looking at a directory
    if os.path.isdir(class_dir):
        # The path to the annoying nested folder
        nested_dir = os.path.join(class_dir, class_folder)
        
        # If the nested folder exists, move everything out of it
        if os.path.exists(nested_dir) and os.path.isdir(nested_dir):
            print(f"Fixing nested folder: {class_folder}...")
            
            # Move all videos up one level
            for file_name in os.listdir(nested_dir):
                old_path = os.path.join(nested_dir, file_name)
                new_path = os.path.join(class_dir, file_name)
                shutil.move(old_path, new_path)
            
            # Delete the empty inner folder
            os.rmdir(nested_dir)

print("Done! All nested folders have been flattened.")