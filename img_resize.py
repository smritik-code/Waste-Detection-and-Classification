import os
import cv2
from tqdm import tqdm

def resize_dataset():
    dataset_path = "trash_dataset"
    output_path = "trash_dataset_320"
    target_size = 320

    print(f"Resizing dataset from 640px to {target_size}px")
    print(f"Input: {dataset_path}")
    print(f"Output: {output_path}")

    # Create output directory structure
    splits = ['train', 'valid', 'test']
    for split in splits:
        os.makedirs(f"{output_path}/{split}/images", exist_ok=True)
        os.makedirs(f"{output_path}/{split}/labels", exist_ok=True)

    # Process each split
    for split in splits:
        print(f"\nProcessing {split} split...")

        input_images_dir = f"{dataset_path}/{split}/images"
        input_labels_dir = f"{dataset_path}/{split}/labels"
        output_images_dir = f"{output_path}/{split}/images"
        output_labels_dir = f"{output_path}/{split}/labels"

        # Get all images
        images = [f for f in os.listdir(input_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for image_name in tqdm(images):
            # Input paths
            input_image_path = os.path.join(input_images_dir, image_name)
            input_label_path = os.path.join(input_labels_dir, image_name.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))

            # Output paths
            output_image_path = os.path.join(output_images_dir, image_name)
            output_label_path = os.path.join(output_labels_dir, image_name.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))

            # Load and resize image
            image = cv2.imread(input_image_path)
            if image is None:
                continue

            resized_image = cv2.resize(image, (target_size, target_size))

            # Save resized image
            cv2.imwrite(output_image_path, resized_image)

            # Copy labels (YOLO format is normalized, so no changes needed)
            if os.path.exists(input_label_path):
                with open(input_label_path, 'r') as f_in:
                    with open(output_label_path, 'w') as f_out:
                        f_out.write(f_in.read())

    # Copy data.yaml
    import shutil
    shutil.copy2(f"{dataset_path}/data.yaml", f"{output_path}/data.yaml")

    print(f"\nResizing complete! New dataset: {output_path}")

if __name__ == "__main__":
    resize_dataset()
