# Waste Detection AI

A deep learning-based object detection system for identifying and classifying recyclable waste materials in images. Built with YOLOv8, this project demonstrates practical computer vision applications for environmental sustainability.

## Performance Highlights

- **Overall mAP50**: 0.480
- **Best Classes**: Metal (0.686 mAP50), Plastic (0.602 mAP50)
- **Inference Speed**: ~80ms on CPU
- **Model Size**: 6.2MB (YOLOv8n)

## Features

- **5 Waste Classes**: CARDBOARD, GLASS, METAL, PAPER, PLASTIC
- **Web Interface**: User-friendly Gradio application
- **Auto-Resize**: Handles any image size automatically
- **Confidence Tuning**: Adjustable detection sensitivity


### Usage

1. **Upload Image**: Drag and drop any waste image
2. **Adjust Confidence**: Use slider (0.1-0.9) to control detection sensitivity
3. **View Results**: See detected objects with bounding boxes and confidence scores
4. **Analyze**: Review detailed statistics and class distributions


## Technical Details

### Model Architecture
- **Backbone**: YOLOv8n (Nano)
- **Input Size**: 320x320 pixels
- **Classes**: 5 recyclable materials
- **Training**: 25 epochs on balanced dataset

### Training Strategy
- **Dataset**: 3,500 training images (balanced)
- **Augmentation**: Mosaic, flip, color adjustments
- **Optimizer**: AdamW with automatic learning rate
- **Hardware**: CPU training (13th Gen Intel i5)


## Dataset

The model was trained on a curated subset of waste detection datasets:
- 3,500 training images
- 1,481 validation images  
- 1,042 test images
- Focused on recyclable materials (excluding biodegradable waste)


## Acknowledgments

- Built with Ultralytics YOLOv8
- Trained on waste detection datasets from Kaggle : https://www.kaggle.com/datasets/viswaprakash1990/garbage-detection


## Demo
<img width="1848" height="988" alt="image" src="https://github.com/user-attachments/assets/922438de-ce10-45ab-acde-e5fd3b7c4653" />
<img width="1851" height="981" alt="image" src="https://github.com/user-attachments/assets/47a05961-9a90-4238-bb23-2483473285c9" />
<img width="1855" height="996" alt="image" src="https://github.com/user-attachments/assets/cb3590a2-8552-4401-b335-c0b46a0b70f4" />


