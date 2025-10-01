from ultralytics import YOLO
import yaml
import os

def evaluate_on_test_set():
    # Load the best model
    model_path = 'runs/detect/waste_50percent2/weights/last.pt'
    model = YOLO(model_path)

    print("OFFICIAL TEST SET EVALUATION")
    print("=" * 60)

    # Run evaluation on test set
    results = model.val(
        data='trash_dataset_final/data.yaml',
        split='test',  # Use the official test set
        save_json=True,
        conf=0.2,
        verbose=True
    )

    print("\n TEST SET RESULTS:")
    print("=" * 60)

    # Print class-wise performance
    class_names = ['CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']

    if hasattr(results, 'box') and results.box:
        print(f"Overall mAP50: {results.box.map50:.3f}")
        print(f"Overall mAP50-95: {results.box.map:.3f}")
        print(f"Precision: {results.box.mp:.3f}")
        print(f"Recall: {results.box.mr:.3f}")

        print("\n CLASS-WISE PERFORMANCE:")
        print("-" * 40)

        if hasattr(results.box, 'ap_class_index'):
            for i, class_idx in enumerate(results.box.ap_class_index):
                class_name = class_names[class_idx]
                ap50 = results.box.ap50[i]
                ap = results.box.ap[i]
                precision = results.box.p[i]
                recall = results.box.r[i]

                print(f"{class_name:12} | mAP50: {ap50:.3f} | mAP: {ap:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")

def analyze_test_predictions():
    # Load model
    model = YOLO('runs/detect/waste_50percent2/weights/last.pt')

    print("\n SAMPLE TEST SET PREDICTIONS")
    print("=" * 60)

    # Test set path from data.yaml
    with open('trash_dataset_final/data.yaml', 'r') as f:
        data_config = yaml.safe_load(f)

    test_images_dir = os.path.join(data_config['path'], data_config['test'])

    # Get first 5 test images
    test_images = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir)
                   if f.endswith(('.jpg', '.jpeg', '.png'))][:5]

    for i, image_path in enumerate(test_images):
        results = model.predict(image_path, conf=0.25, verbose=False)
        result = results[0]

        print(f"\n{i+1}. {os.path.basename(image_path)}")
        print(f"   Detected {len(result.boxes)} objects:")

        if len(result.boxes) > 0:
            class_counts = {}
            for box in result.boxes:
                class_name = model.names[int(box.cls[0])]
                confidence = float(box.conf[0])

                if class_name not in class_counts:
                    class_counts[class_name] = []
                class_counts[class_name].append(confidence)

            for class_name, confidences in class_counts.items():
                avg_conf = sum(confidences) / len(confidences)
                print(f"     ▸ {class_name}: {len(confidences)} (avg conf: {avg_conf:.3f})")
        else:
            print("  No objects detected")

if __name__ == "__main__":
    evaluate_on_test_set()
    analyze_test_predictions()
