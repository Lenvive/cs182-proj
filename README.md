# Emotion Recognition with Attention Mechanisms

This project implements emotion recognition using CNN models with attention mechanisms and compares performance with traditional PCA+SVM approaches.

## Project Overview

The project includes:
- Two CNN models: full image processing and attention-based cropped region processing
- Grad-CAM visualization for attention analysis
- PCA+SVM: Traditional methods 
- Comprehensive performance evaluation and visualization

## System Requirements

- **CUDA Version**: 12.1
- **Python**: 3.10


## Environment Setup


```bash
# Create a new conda environment
conda create -n emotion_recognition python=3.10 -y
# Activate the environment
conda activate emotion_recognition
```



```bash
# Install PyTorch with CUDA 12.1 (must be done first, and the cuda version on my computer is 12.1, you can modify this according to your device)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```


```bash
pip install -r requirements.txt
```


## Project Structure

```
cs182-proj/
├── README.md
├── requirements.txt
├── src/
│   └── emotion_recognition_attetion.ipynb
├── data/
│  
├── models/ (generated during training)
│   ├── first_model.pth
│   ├── second_model.pth
│   └── pca_svm_model_full.joblib
└── results/ (generated during execution)
```

## Data Preparation

1. **Download the FER2013 dataset** or use your emotion recognition dataset
2. **Organize data** into the following structure:
   ```
   data/
   ├── train/
   │   ├── angry/
   │   ├── disgust/
   │   ├── fear/
   │   ├── happy/
   │   ├── neutral/
   │   ├── sad/
   │   └── surprise/
   └── test/
       ├── angry/
       ├── disgust/
       ├── fear/
       ├── happy/
       ├── neutral/
       ├── sad/
       └── surprise/
   ```
3. **Image format**: PNG files, preferably 48x48 pixels (will be resized automatically)

## Usage Instructions

### 1. Start Jupyter Notebook

```bash
# Activate environment
conda activate emotion_recognition

# Start Jupyter
jupyter notebook
```

### 2. Open the Main Notebook

Open `src/emotion_recognition_attetion.ipynb` in Jupyter and select the "Python (Emotion Recognition)" kernel.

### 3. Run the Analysis

The notebook includes the following main sections:

1. **Data Loading and Preprocessing**
   - Custom dataset class for emotion images
   - Data augmentation and normalization

2. **First CNN Model**
   - Standard CNN architecture for full image processing
   - Training with early stopping and learning rate scheduling

3. **Grad-CAM Visualization**
   - Attention mechanism visualization
   - Important region identification

4. **Second CNN Model**
   - Attention-based model using cropped important regions
   - Specialized architecture for smaller input size

5. **PCA+SVM Baseline**
   - Traditional machine learning approach
   - Dimensionality reduction with PCA
   - Support Vector Machine classification

6. **Comprehensive Evaluation**
   - Performance comparison across all models
   - Confusion matrices and classification reports
   - Visualization of results

### 4. Model Training

Models will be automatically saved and loaded:
- `first_model.pth`: First CNN model weights
- `second_model.pth`: Second CNN model weights  
- `pca_svm_model_full.joblib`: Complete PCA+SVM pipeline

## Expected Results

The project typically achieves:
- **First CNN**: ~60-70% accuracy on emotion recognition
- **Second CNN (Attention)**: Similar or improved accuracy with interpretability
- **PCA+SVM**: Baseline performance for comparison






### Environment Conflicts

If you encounter environment conflicts:

```bash
# Remove environment and recreate
conda deactivate
conda env remove -n emotion_recognition
# Then follow setup steps again
```


