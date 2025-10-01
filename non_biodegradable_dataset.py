import os
import shutil

def create_non_biodegradable_dataset():
    original_path = "trash_dataset_320"
    focused_path = "trash_dataset_final"

    # Remove BIODEGRADABLE (class 0), keep others
    classes_to_keep = [1, 2, 3, 4, 5]  # CARDBOARD, GLASS, METAL, PAPER, PLASTIC
    class_names = ['CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']

    # Create directory structure
    for split in ['train', 'valid', 'test']:
        os.makedirs(f"{focused_path}/{split}/images", exist_ok=True)
        os.makedirs(f"{focused_path}/{split}/labels", exist_ok=True)

    # Process each split
    for split in ['train', 'valid', 'test']:
        images_dir = f"{original_path}/{split}/images"
        labels_dir = f"{original_path}/{split}/labels"

        all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        kept_images = 0

        for img_name in all_images:
            label_name = img_name.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
            label_path = f"{labels_dir}/{label_name}"

            if not os.path.exists(label_path):
                continue

            # Read labels and remove BIODEGRADABLE objects
            new_labels = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        # Only keep non-biodegradable classes
                        if class_id in classes_to_keep:
                            # Remap class IDs (1→0, 2→1, 3→2, 4→3, 5→4)
                            new_class_id = classes_to_keep.index(class_id)
                            new_line = f"{new_class_id} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n"
                            new_labels.append(new_line)

            # Only copy images that have non-biodegradable objects
            if new_labels:
                # Copy image
                shutil.copy2(f"{images_dir}/{img_name}", f"{focused_path}/{split}/images/{img_name}")

                # Save filtered labels
                with open(f"{focused_path}/{split}/labels/{label_name}", 'w') as f:
                    f.writelines(new_labels)

                kept_images += 1

        print(f"{split}: Kept {kept_images}/{len(all_images)} images (with non-biodegradable objects)")

    # Create new data.yaml with 5 classes
    yaml_content = f"""path: {os.path.abspath(focused_path)}
train: train/images
val: valid/images
test: test/images

nc: 5
names: {class_names}

imgsz: 320
"""
    with open(f"{focused_path}/data.yaml", 'w') as f:
        f.write(yaml_content)

    print(f"Recycling-focused dataset created: {focused_path}")
    print(f"Classes: {class_names}")

create_non_biodegradable_dataset()
