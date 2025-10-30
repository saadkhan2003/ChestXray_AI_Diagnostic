# Models Folder

This folder contains your trained model files.

## 📦 Your Models

- `pneumonia_model_best.h5` - Pneumonia detection model
- `covid_model_best.h5` - COVID-19 detection model
- Other .h5 or .keras files (backups/variants)

## ⚠️ Important Notes

### For Local Storage
- ✅ Keep these files safe locally
- ✅ Back them up (they took 5 hours to train!)
- ✅ You'll need them for deployment

### For GitHub
- ❌ These files are **gitignored** (too large for GitHub)
- ❌ They will NOT be pushed to your repository
- ✅ This is normal and correct!

### For Deployment
When deploying on Colab:
1. Upload these .h5 files to Colab session storage
2. Place them in `/content/models/` folder
3. The notebook will load them from there
4. See deployment guides for detailed steps

## 📊 Model Information

### Pneumonia Model
- **Architecture**: DenseNet-121 with transfer learning
- **Input**: 224x224 RGB chest X-ray images
- **Output**: Binary classification (Normal vs Pneumonia)
- **File Size**: ~80-100 MB

### COVID-19 Model
- **Architecture**: DenseNet-121 with transfer learning
- **Input**: 224x224 RGB chest X-ray images
- **Output**: Binary classification (Normal vs COVID-19)
- **File Size**: ~80-100 MB

## 🚀 Usage

### In Colab Notebook
```python
# Models are loaded like this in the notebook:
pneumonia_model = tf.keras.models.load_model(
    '/content/models/pneumonia_model_best.h5',
    compile=False
)

covid_model = tf.keras.models.load_model(
    '/content/models/covid_model_best.h5',
    compile=False
)
```

### Local Testing (if needed)
```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('models/pneumonia_model_best.h5')

# Make prediction
prediction = model.predict(preprocessed_image)
```

## 📁 File Structure

```
models/
├── .gitkeep                        # For git tracking (empty folder)
├── README.md                       # This file
├── pneumonia_model_best.h5         # Main pneumonia model (gitignored)
├── covid_model_best.h5             # Main COVID model (gitignored)
└── [other model variants]          # Backups/experiments (gitignored)
```

## 🔒 Security Note

These models are for educational use only. Do not use for actual medical diagnosis.

---

**Last Updated**: October 30, 2025  
**Total Models**: 2 main models (Pneumonia, COVID-19)  
**Status**: Ready for deployment
