import os
import argparse

def create_image_pairs(image_folder: str, output_file: str = "image_pairs.txt", num_matches: int = 6):
    # Get the list of image files in the folder, sorted alphabetically
    image_files = sorted(
        [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    )

    if not image_files:
        print("No valid image files found in the specified folder.")
        return

    # Open the output file in write mode
    with open(output_file, "w") as f:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create image pairs from a folder of images.")
    parser.add_argument('--image_folder', type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument('--output_file', type=str, default="image_pairs.txt", help="Path to the output file.")
    parser.add_argument('--num_matches', type=int, default=6, help="Number of matches to create for each image.")
    args = parser.parse_args()

    create_image_pairs(args.image_folder, args.output_file, args.num_matches)
