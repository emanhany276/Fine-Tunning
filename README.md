### Teeth Classification AI

### Overview
This project aims to classify different types of teeth using deep learning. It utilizes a pre-trained ResNet50 model fine-tuned on a custom dataset to recognize seven categories of teeth conditions. The model is integrated into a Streamlit web application, allowing users to upload images and receive predictions.

### Features
- **Deep Learning-Based Classification**: Uses ResNet50 with transfer learning.
- **Streamlit Web Interface**: User-friendly interface for image uploads and predictions.
- **Data Augmentation**: Improves model generalization.
- **Fine-Tuning**: Enhances model accuracy through additional training.
- **Visualization**: Plots training and validation accuracy/loss trends.

### Dataset
The dataset is structured into three sets:
- **Training Set**: Used for model training.
- **Validation Set**: Used for hyperparameter tuning.
- **Testing Set**: Used for final evaluation.

Dataset location:
```
Teeth_Dataset/
│── Training/
│── Validation/
│── Testing/
```
Each folder contains subfolders corresponding to class labels.

### Model Architecture
- **Base Model**: ResNet50 (pre-trained on ImageNet, without top layers)
- **Additional Layers**:
  - Global Average Pooling
  - Dropout (0.5 for regularization)
  - Dense layer with softmax activation (7 classes)

### Installation & Setup
#### Prerequisites
Ensure you have Python 3.7+ and install dependencies:
```bash
pip install tensorflow streamlit numpy matplotlib pillow
```

#### Running the Training Script
Train the model by executing:
```bash
python train.py
```

#### Running the Streamlit App
To start the web interface:
```bash
streamlit run app.py
```

### Model Training Process
1. **Data Loading & Preprocessing**: Normalizes images for ResNet50.
2. **Initial Training**: Trains a frozen ResNet50 model.
3. **Fine-Tuning**: Unfreezes top layers for further training.
4. **Evaluation**: Tests model performance.

### Performance Evaluation
- Model accuracy is monitored during training.
- Fine-tuning improves classification results.
- Accuracy and loss plots are visualized using Matplotlib.

### Usage
1. Launch the Streamlit app.
2. Upload an image of a tooth.
3. The model predicts the class with confidence scores.

### Class Labels
- `Cas` (Caries)
- `Cos` (Cosmetic Issue)
- `Gum` (Gum Disease)
- `MC` (Missing Crown)
- `OC` (Orthodontic Condition)
- `OLP` (Oral Lichen Planus)
- `OT` (Other Condition)

