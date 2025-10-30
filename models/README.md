# Models Directory

This directory will contain the trained models for disease detection.

## Expected Models:

1. **pneumonia_model.h5** - Pneumonia detection model
2. **covid_model.h5** - COVID-19 detection model
3. **lung_cancer_model.h5** - Lung cancer detection model

## File Structure:
```
models/
├── pneumonia_model.h5
├── covid_model.h5
├── lung_cancer_model.h5
├── model_metadata.json (optional)
└── README.md (this file)
```

## Training:

Models will be trained using the Jupyter notebooks in the `notebooks/` directory.

After training on Google Colab:
1. Download the `.h5` model files
2. Place them in this directory
3. Run the Streamlit app

## Model Details:

### Pneumonia Model
- **Architecture**: DenseNet-121 (pre-trained + fine-tuned)
- **Input**: 224x224x3 RGB images
- **Output**: 2 classes (Normal, Pneumonia)
- **Expected Size**: ~30-40 MB

### COVID-19 Model
- **Architecture**: DenseNet-121 (pre-trained + fine-tuned)
- **Input**: 224x224x3 RGB images
- **Output**: 2 classes (Normal, COVID-19)
- **Expected Size**: ~30-40 MB

### Lung Cancer Model
- **Architecture**: DenseNet-121 (pre-trained + fine-tuned)
- **Input**: 224x224x3 RGB images
- **Output**: 3 classes (Normal, Benign, Malignant)
- **Expected Size**: ~30-40 MB

---

**Note**: Model files are excluded from git due to their large size (see `.gitignore`).
