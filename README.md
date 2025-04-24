# Rice Image Classification System

This project implements an image classification system to accurately identify different types of rice grains using Convolutional Neural Networks (CNN) with transfer learning. The system includes a user interface for uploading images and receiving predictions.

## Project Overview

The goal of this project is to build a deep learning model that can classify rice grains into their respective varieties. The model uses transfer learning with a pre-trained VGG16 network and achieves high accuracy in distinguishing between different rice types.

### Features

- Transfer learning with VGG16 architecture
- Data augmentation to enhance model generalization
- Two-phase training approach (frozen base model, then fine-tuning)
- Comprehensive evaluation metrics (accuracy, precision, recall, F1-score)
- Interactive UI for image upload and prediction

## Dataset

The project uses the [Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset) from Kaggle, which contains images of 5 different rice varieties:
- Arborio
- Basmati
- Ipsala
- Jasmine
- Karacadag

## Environment Setup

### Prerequisites

- Python 3.8+
- Google Colab (recommended) or local Python environment
- GPU access (recommended for faster training)

### Dependencies

```
tensorflow>=2.8.0
numpy>=1.19.5
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
seaborn>=0.11.0
gradio>=3.0.0
kagglehub (if downloading from Kaggle)
```

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/rice-classification.git
   cd rice-classification
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download the dataset:
   - Option 1: Download from [Kaggle](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset)
   - Option 2: Use kagglehub in the notebook

## Usage

### Training the Model

1. Open the Jupyter notebook or Google Colab:
   ```
   jupyter notebook Rice_Classification.ipynb
   ```

2. Run the cells sequentially to:
   - Load and preprocess the dataset
   - Create data generators with augmentation
   - Build and train the model
   - Evaluate performance and generate visualizations

### Using the UI

After training the model, you can:

1. Run the UI cells to start the Gradio interface
2. Upload an image of rice grains
3. View prediction results showing the rice variety and confidence scores

## Project Structure

```
rice-classification/
├── Rice_Classification.ipynb  # Main notebook with all code
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
├── rice_classifier_final.h5   # Saved model (after training)
└── screenshots/               # Application screenshots
```

## Performance Metrics

The model achieves the following performance on the test set:
- Accuracy: ~99%
- Precision: ~98%
- Recall: ~99%
- F1 Score: ~98%
- Test Loss: 0.0160

## Model Architecture

The model uses transfer learning with VGG16 as the base model, with additional layers:
- Global average pooling
- Dense layers (512 → 256 → 5 output classes)
- Dropout and batch normalization for regularization

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset) by Murat Koklu
- TensorFlow and Keras documentation
- Gradio for the UI implementation
