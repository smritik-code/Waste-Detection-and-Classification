import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd

# Load model
model = YOLO('runs/detect/waste_50percent2/weights/best.pt')

# Color mapping
colors = {
    'CARDBOARD': (56, 142, 60),    # Green
    'GLASS': (2, 136, 209),        # Blue
    'METAL': (245, 124, 0),        # Orange
    'PAPER': (156, 39, 176),       # Purple
    'PLASTIC': (229, 57, 53)       # Red
}

def resize_for_detection(image, target_size=320):
    """Auto-resize image to target size for detection"""
    h, w = image.shape[:2]

    # If already 320x320, return as is
    if h == target_size and w == target_size:
        return image, f"✅ Perfect size: {w}x{h}"

    # Calculate scaling factor
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create square canvas with padding
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)

    # Calculate padding and center image
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

    resize_info = f"🔄 Resized from {w}x{h} to 320x320"
    return canvas, resize_info

def detect_waste(image, confidence_threshold, show_stats):
    """Detection with automatic background resizing"""

    # Auto-resize in background (no user choice needed)
    processed_image, resize_info = resize_for_detection(image)

    # Run detection
    results = model.predict(processed_image, imgsz=320, conf=confidence_threshold, verbose=False)
    result = results[0]

    # Create output image
    output_image = processed_image.copy()
    detections = []
    class_stats = {}

    if len(result.boxes) > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[class_id]

            # Get color
            color = colors.get(class_name, (0, 0, 0))

            # Draw bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 3)    # Resize image


            # Draw label with background
            label = f"{class_name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(output_image, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
            cv2.putText(output_image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })

            # Update stats
            if class_name not in class_stats:
                class_stats[class_name] = []
            class_stats[class_name].append(confidence)

    # Create results text
    results_text = "## 🔍 Detection Results\n\n"
    results_text += f"{resize_info}\n\n"

    if detections:
        # Summary
        results_text += f"**📦 Total Objects Detected:** {len(detections)}\n\n"

        results_text += "### 📊 Object Details:\n"
        for i, det in enumerate(sorted(detections, key=lambda x: x['confidence'], reverse=True), 1):
            confidence_emoji = "🟢" if det['confidence'] > 0.7 else "🟡" if det['confidence'] > 0.5 else "🟠"
            results_text += f"{confidence_emoji} **{det['class']}** - Confidence: `{det['confidence']:.3f}`<br>"

        # Statistics
        if show_stats and class_stats:
            results_text += "\n### 📈 Class Statistics:\n"
            for class_name, confidences in class_stats.items():
                avg_conf = sum(confidences) / len(confidences)
                results_text += f"• **{class_name}**: {len(confidences)} objects (avg: `{avg_conf:.3f}`)\n"

        # Performance info
        results_text += f"\n**⚡ Inference Speed:** {result.speed['inference']:.1f}ms\n"
        results_text += f"**🎯 Confidence Threshold:** {confidence_threshold}"

    else:
        results_text += "❌ **No objects detected**\n\n"
        results_text += "*Try lowering the confidence threshold or using a different image.*"

    return output_image, results_text

def show_model_metrics():
    metrics = {
        'Class': ['METAL', 'PLASTIC', 'PAPER', 'CARDBOARD', 'GLASS'],
        'mAP50': [0.686, 0.602, 0.569, 0.061, 0.000],
        'Precision': [0.707, 0.673, 0.791, 0.028, 0.000],
        'Recall': [0.597, 0.485, 0.335, 0.350, 0.000]
    }

    df = pd.DataFrame(metrics)
    metrics_text = "## 🎯 Model Performance\n\n"
    metrics_text += "**Overall mAP50: 0.480**\n\n"

    for _, row in df.iterrows():
        metrics_text += f"**{row['Class']}:** mAP50=`{row['mAP50']:.3f}` | Precision=`{row['Precision']:.3f}` | Recall=`{row['Recall']:.3f}`\n"

    return metrics_text

# Create the interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    # Header
    gr.Markdown("# 🚮 Waste Detection AI")
    gr.Markdown("### Upload any image - automatically resized to 320px for optimal detection")

    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            gr.Markdown("## 📤 Input Settings")

            image_input = gr.Image(
                label="Upload Waste Image (Any Size)",
                type="numpy",
                height=300
            )

            confidence_slider = gr.Slider(
                label="Confidence Threshold",
                minimum=0.1,
                maximum=0.9,
                value=0.3,
                step=0.05
            )

            show_stats = gr.Checkbox(
                label="Show Detailed Statistics",
                value=True
            )

            detect_btn = gr.Button("🔍 Detect Waste", variant="primary", size="lg")

            # Model info
            gr.Markdown("## ℹ️ Model Info")
            gr.Markdown("""
            - **Model**: YOLOv8 Nano
            - **Auto-Resize**: All images → 320×320 pixels
            - **Classes**: 5 recyclable materials
            - **Performance**: 0.48 mAP50
            - **Best Classes**: Metal (0.69), Plastic (0.60)
            """)

        with gr.Column(scale=1):
            # Output section
            gr.Markdown("## 📊 Detection Results")

            image_output = gr.Image(
                label="Processed Image (320×320)",
                type="numpy",
                height=400
            )

            results_output = gr.Markdown(
                label="Analysis Results",
                value="## 🔍 Detection Results\n\n*Upload an image and click detect to see results*"
            )

    # Examples
    gr.Markdown("## 🎯 Try These Examples")

    with gr.Row():
        gr.Examples(
            examples=[
                "custom_test_images/glass.jpg",
                "custom_test_images/waste-paper-products-cardboard-box-13371822.jpg",
                "custom_test_images/metal.jpg"
            ],
            inputs=image_input
        )


    # Connect the button
    detect_btn.click(
        fn=detect_waste,
        inputs=[image_input, confidence_slider, show_stats],
        outputs=[image_output, results_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        share=False
    )
