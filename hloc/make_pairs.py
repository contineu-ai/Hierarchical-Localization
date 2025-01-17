import os

def create_image_pairs(image_folder: str, output_file: str = "image_pairs.txt", num_matches: int = 6):
    # Get the list of image files in the folder, sorted alphabetically
    image_files = sorted(
        [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    )

    # Open the output file in write mode
    with open(output_file, "a") as f:
        # Loop through each image in the folder
        for i in range(len(image_files)):
            # For each image, pair it with the next `num_matches` images
            for j in range(1, num_matches + 1):
                if i + j < len(image_files):  # Ensure we don't go out of bounds
                    img1_path = image_files[i]
                    img2_path = image_files[i + j]
                    f.write(f"{img1_path} {img2_path}\n")
                else:
                    break  # If there are not enough next images, stop

    print(f"Image pairs written to {output_file}")

# Example usage:
image_folder = "/data/sahil/data/sahil_test_videos/6759a65ed5305e001214c976"
create_image_pairs(image_folder)