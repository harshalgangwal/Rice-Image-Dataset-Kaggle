# Rice Grain Classification System

This project implements an image classification system that can accurately classify different types of rice grains using a Convolutional Neural Network (CNN). The system utilizes transfer learning with a pre-trained VGG16 model to classify rice grains into five different categories: Arborio, Basmati, Ipsala, Jasmine, and Karacadag.

## Project Structure

- `data_preparation.py`: Handles dataset loading, exploration, and preprocessing
- `model_training.py`: Defines and trains the CNN model using transfer learning
- `model_evaluation.py`: Evaluates model performance with metrics and visualizations
- `app.py`: Streamlit application for user interface
- `rice_classifier_model.h5`: Saved trained model

## Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/rice-classification-system.git
cd rice-classification-system
```

2. Install the required packages:
```
pip install -r requirements.txt
```

3. Download the rice image dataset from Kaggle:
```python
import kagglehub
rice_image_dataset = kagglehub.dataset_download('muratkokludataset/rice-image-dataset')
```

### Running the Application

1. Launch the Streamlit web application:
```
streamlit run app.py
```

2. Open your web browser and go to `http://localhost:8501`

## Model Architecture

The model architecture consists of:
- VGG16 base model (pre-trained on ImageNet)
- Custom classification head with:
  - Dense layer (256 neurons)
  - BatchNormalization layer
  - Dropout layer (0.5)
  - Output layer with softmax activation

## Model Performance

- Accuracy: ~95%
- Precision: ~95%
- Recall: ~94% 
- F1-Score: ~94%

## Usage

1. Upload an image of rice grains using the web interface
2. Click "Classify Rice Type"
3. View the prediction and probability distribution

## Dependencies

- TensorFlow 2.x
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Pillow (PIL)

## Dataset

The Rice Image Dataset contains images of five rice grain varieties:
- Arborio
- Basmati
- Ipsala
- Jasmine
- Karacadag

Dataset source: [Rice Image Dataset on Kaggle](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)
