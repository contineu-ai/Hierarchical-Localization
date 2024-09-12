import os
from PIL import Image

def resize_images(input_folder, output_folder, size):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Open the image
            with Image.open(os.path.join(input_folder, filename)) as img:
                # Resize the image
                resized_img = img.resize(size)
                
                # Save the resized image to the output folder
                output_path = os.path.join(output_folder, filename)
                resized_img.save(output_path)
                print(f"Resized and saved: {filename}")

# Example usage
input_folder = "/home/luffy/data/data_raghuvir/VID_20240622_155518_00_007_processed"
output_folder = "/home/luffy/data/data_raghuvir/VID_20240622_155518_00_007_processed_resized"
new_size = (3840, 1920)  # Width, Height

resize_images(input_folder, output_folder, new_size)