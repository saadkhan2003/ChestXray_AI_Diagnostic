# 🫁 Chest X-ray Disease Detection System

**AI-Powered Medical Image Analysis using Deep Learning + Gradio Web Interface**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.8+](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://tensorflow.org)
[![Gradio](https://img.shields.io/badge/Gradio-3.0+-green.svg)](https://gradio.app)

---

## 📋 Project Overview

This project implements an AI-powered chest X-ray disease detection system that uses deep learning to identify respiratory conditions. The system is deployed as an interactive web application using Gradio on Google Colab, making it accessible and easy to use.

### 🎯 Diseases Detected
- **Pneumonia** (Normal vs Pneumonia classification)
- **COVID-19** (Normal vs COVID-19 classification)

### 🌟 Key Features
- 🤖 **Transfer Learning** with DenseNet-121 pre-trained architecture
- 🎨 **Interactive Web Interface** built with Gradio
- ☁️ **Google Colab Deployment** with free GPU support
- 🌐 **Shareable Public URLs** for easy access
- 📱 **Mobile-Friendly** responsive design
- ⚡ **Fast Predictions** - results in 2-3 seconds
- 📊 **Visual Confidence Scores** with bar charts
- � **Multi-Model Analysis** - uses both Pneumonia and COVID-19 models

---

## 🎯 Project Status

✅ **Complete and Functional**
- Working Jupyter notebook with Gradio deployment
- Pre-trained model loading system
- Interactive web interface
- Real-time prediction capabilities
- Ready for demonstration and deployment

---

## 🚀 Quick Start Guide

### Prerequisites
- Google account (for Google Colab)
- Trained model files (`pneumonia_model_best.h5` and `covid_model_best.h5`)
- Basic understanding of Jupyter notebooks

### Deployment Steps

1. **Open the Notebook in Google Colab**
   - Upload `notebooks/XRAY_Detection.ipynb` to [Google Colab](https://colab.research.google.com/)
   - Or open directly from GitHub

2. **Enable GPU Runtime**
   ```
   Runtime → Change runtime type → Hardware accelerator: GPU → Save
   ```

3. **Mount Google Drive**
   - Run the first cells to mount your Google Drive
   - Upload your trained models to Google Drive
   - Update the `PROJECT_DIR` path in the notebook to point to your models

4. **Install Dependencies**
   - The notebook will automatically install required packages (Gradio, etc.)

5. **Run All Cells**
   - Execute cells sequentially or use `Runtime → Run all`
   - Models will be loaded from your specified Google Drive location

6. **Get Your Public URL**
   - Once launched, Gradio generates a shareable public link
   - Share this URL with anyone to access your app
   - Link remains active as long as the Colab session is running

**That's it!** Your AI diagnostic tool is now live! 🎉

---

## 📁 Project Structure

```
ChestXray_AI_Diagnostic/
│
├── notebooks/
│   └── XRAY_Detection.ipynb            # ⭐ Main notebook - Gradio deployment
│
├── models/
│   ├── README.md                       # Model information
│   ├── README_MODELS.md                # Detailed model documentation
│   └── [.h5 files]                     # Your trained models (gitignored)
│
├── datasets/
│   └── README.md                       # Dataset download instructions
│
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore configuration
└── README.md                           # This file
```

### File Descriptions

- **`XRAY_Detection.ipynb`**: Complete notebook with model loading, Gradio interface, and prediction functions
- **`requirements.txt`**: All Python package dependencies for the project
- **`models/`**: Directory for storing trained .h5 model files (excluded from git due to size)
- **`datasets/`**: Instructions for downloading and organizing training datasets

---

## 🎨 Web Interface Features

### Gradio Application
The deployed application provides:

- 📤 **Drag & Drop Upload** - Easy image upload interface
- 🔍 **Real-time Analysis** - Instant predictions (2-3 seconds)
- 📊 **Dual Model Prediction** - Results from both Pneumonia and COVID-19 models
- 📈 **Visual Confidence Scores** - Interactive bar charts showing prediction confidence
- 🌐 **Public Shareable URL** - Access from anywhere with an internet connection
- 📱 **Responsive Design** - Works seamlessly on desktop and mobile devices
- 💡 **Clear Diagnosis Display** - Color-coded results with confidence percentages

### Example Output

When you upload a chest X-ray image, the system provides:

```
✅ Diagnosis Status:
   "⚠️ PNEUMONIA DETECTED (Confidence: 87.3%)"
   
📊 Detailed Confidence Scores:
   ├─ Normal (Pneumonia Model): 12.7%
   ├─ Pneumonia:               87.3%
   ├─ Normal (COVID Model):    94.2%
   └─ COVID-19:                 5.8%

📈 [Interactive Bar Chart Visualization]
```

---

## 🧪 Model Architecture & Technology

### Deep Learning Model

**Architecture**: DenseNet-121 with Transfer Learning

The models use the DenseNet-121 architecture pre-trained on ImageNet, which provides:
- Dense connectivity between layers for efficient feature propagation
- Reduced number of parameters compared to traditional CNNs
- Strong performance on medical imaging tasks

### Model Specifications

**Pneumonia Detection Model**
- **Input**: 224×224×3 RGB chest X-ray images
- **Output**: Binary classification (Normal vs Pneumonia)
- **Architecture**: DenseNet-121 base + custom classification head
- **File Size**: ~80-100 MB

**COVID-19 Detection Model**
- **Input**: 224×224×3 RGB chest X-ray images
- **Output**: Binary classification (Normal vs COVID-19)
- **Architecture**: DenseNet-121 base + custom classification head
- **File Size**: ~80-100 MB

### Image Preprocessing

```python
1. Convert to RGB (if needed)
2. Resize to 224×224 pixels
3. Normalize pixel values (0-1 range)
4. Batch dimension expansion
```

---

## 📊 Datasets

### Training Data Sources

The models in this project are trained on publicly available medical imaging datasets:

#### 1. Pneumonia Detection Dataset
- **Source**: [Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) on Kaggle
- **Classes**: Normal, Pneumonia
- **Total Images**: ~5,863 X-ray images
- **Structure**: Organized into train/validation/test splits with separate folders for each class

#### 2. COVID-19 Detection Dataset
- **Source**: [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) on Kaggle
- **Classes**: Normal, COVID-19
- **Total Images**: ~21,165 X-ray images
- **Structure**: Organized dataset with Normal and COVID-19 categories

### Dataset Organization

To train your own models, download the datasets and organize them as described in `datasets/README.md`:

```
datasets/
├── pneumonia/
│   └── chest_xray/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       ├── val/
│       └── test/
│
└── covid_organized/
    ├── train/
    │   ├── Normal/
    │   └── COVID/
    ├── val/
    └── test/
```

---

## 💻 Technical Stack

### Core Technologies

| Technology | Purpose | Version |
|------------|---------|---------|
| **TensorFlow/Keras** | Deep learning framework | 2.8.0+ |
| **DenseNet-121** | Pre-trained CNN architecture | ImageNet weights |
| **Gradio** | Web interface framework | 3.0+ |
| **Google Colab** | Cloud deployment platform | Latest |
| **Python** | Programming language | 3.8+ |

### Python Dependencies

The complete list of dependencies is in `requirements.txt`:

```python
# Core ML/DL
tensorflow>=2.8.0
keras>=2.8.0

# Web Interface
gradio>=3.0.0

# Image Processing
opencv-python-headless>=4.5.0
Pillow>=9.0.0

# Data Science
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0

# ML Utilities
scikit-learn>=1.0.0
```

### System Requirements

**For Deployment (Google Colab)**:
- Google account
- Web browser
- Internet connection
- Recommended: GPU runtime for faster predictions

**For Local Development**:
- Python 3.8 or higher
- 8GB+ RAM recommended
- CUDA-compatible GPU (optional, for training)

---

## 🔧 How It Works

### Workflow Overview

```
1. User uploads chest X-ray image via Gradio interface
                    ↓
2. Image preprocessing (resize to 224×224, normalize)
                    ↓
3. Prediction with Pneumonia model
                    ↓
4. Prediction with COVID-19 model
                    ↓
5. Confidence score calculation
                    ↓
6. Results display with visualization
```

### Prediction Logic

The system uses a dual-model approach:

1. **Image Upload**: User uploads a chest X-ray image through the web interface
2. **Preprocessing**: Image is converted to RGB, resized to 224×224, and normalized
3. **Dual Prediction**: 
   - Pneumonia model predicts: Normal vs Pneumonia
   - COVID-19 model predicts: Normal vs COVID-19
4. **Confidence Analysis**: Calculates probability scores for each condition
5. **Result Interpretation**:
   - High confidence (>70%): Clear positive detection
   - Medium confidence (50-70%): Possible abnormality
   - Low confidence (<50%): Likely normal
6. **Visualization**: Displays results with color-coded bar charts

### Key Functions in Notebook

```python
preprocess_image(image)      # Prepares image for model input
predict_disease(image)       # Main prediction function
demo.launch(share=True)      # Launches Gradio interface with public URL
```

---

## 🎓 For Academic & Educational Use

### Ideal For

- 🎓 **Final Year Projects (FYP)** - Complete AI/ML project demonstration
- 📚 **Research Projects** - Medical image analysis research
- 👨‍🏫 **Educational Demonstrations** - Teaching AI in healthcare
- 💼 **Portfolio Projects** - Showcasing ML engineering skills

### What Makes This Project Stand Out

1. ✅ **End-to-End Solution** - From model to deployment
2. ✅ **Modern Technology Stack** - TensorFlow, Gradio, Cloud deployment
3. ✅ **Real-world Application** - Healthcare AI use case
4. ✅ **Accessible Interface** - No technical knowledge required for end users
5. ✅ **Cloud Deployment** - Free, scalable, and shareable

### Presentation Tips

**For Project Demonstration**:
1. Start with the problem statement (medical imaging challenges)
2. Explain the AI solution (transfer learning with DenseNet-121)
3. Show the Jupyter notebook structure and code
4. Demonstrate live predictions using the Gradio interface
5. Discuss results, confidence scores, and model performance
6. Highlight the accessibility via public URL
7. Mention limitations and future improvements

**Demo Checklist**:
- [ ] Have 3-5 test X-ray images ready (various conditions)
- [ ] Ensure Colab notebook is running with public URL active
- [ ] Prepare backup screenshots in case of connectivity issues
- [ ] Know your model accuracy/performance metrics
- [ ] Be ready to explain transfer learning concept
- [ ] Prepare to discuss medical disclaimer and limitations

---

## ⚠️ Important Disclaimers

### Medical Disclaimer

**🚨 CRITICAL: This project is for EDUCATIONAL and RESEARCH purposes ONLY.**

- ❌ **NOT** a medical diagnostic tool
- ❌ **NOT** intended for clinical use
- ❌ **NOT** a replacement for professional medical advice
- ❌ **NOT** validated for actual patient diagnosis
- ✅ Educational demonstration of machine learning concepts
- ✅ Research and learning tool for AI/ML students
- ✅ Portfolio project for software engineering

**⚕️ Always consult qualified healthcare professionals for medical diagnosis and treatment.**

### Ethical Considerations

- This tool is a proof-of-concept for educational purposes
- Real medical AI systems require rigorous validation and regulatory approval
- Healthcare professionals should always be involved in patient diagnosis
- AI should assist, not replace, medical expertise

---

## 🐛 Troubleshooting Guide

### Common Issues & Solutions

#### Model Loading Issues

**Problem**: Error loading models
```
Solution:
1. Verify model file names match exactly:
   - pneumonia_model_best.h5
   - covid_model_best.h5
2. Check the PROJECT_DIR path in the notebook points to correct location
3. Ensure models are in Google Drive in the specified folder
4. Verify file sizes (should be ~80-100MB each)
5. Try re-uploading model files to Google Drive
```

#### Google Drive Mounting Issues

**Problem**: Drive mount fails or shows wrong directory
```
Solution:
1. Run the drive mount cell again
2. Grant permissions when prompted
3. Update PROJECT_DIR variable to match your Drive structure
4. Verify path with: !ls /content/drive/MyDrive/
```

#### Gradio Public URL Issues

**Problem**: Public URL not working or expired
```
Solution:
1. Keep the Colab tab/window open
2. Don't interrupt the cell execution
3. Re-run the launch cell if link expires
4. Check internet connection
5. Try demo.launch(share=True) again
```

#### Performance Issues

**Problem**: Slow predictions or timeout errors
```
Solution:
1. Enable GPU runtime:
   Runtime → Change runtime type → GPU → Save
2. Restart runtime: Runtime → Restart runtime
3. Reduce image size (<5MB recommended)
4. Upload one image at a time
```

#### Memory Errors

**Problem**: "Out of memory" error in Colab
```
Solution:
1. Runtime → Restart runtime
2. Clear output: Edit → Clear all outputs
3. Run cells sequentially (not all at once)
4. Ensure GPU is enabled
5. Close other Colab notebooks
```

#### Import Errors

**Problem**: Module not found errors
```
Solution:
1. Re-run the package installation cell
2. Restart runtime if needed
3. Verify internet connectivity in Colab
4. Check requirements.txt for correct versions
```

---

## 📦 Repository Structure Details

### Core Files

| File/Folder | Description | Size |
|------------|-------------|------|
| `notebooks/XRAY_Detection.ipynb` | Main deployment notebook | ~50KB |
| `requirements.txt` | Python dependencies | ~1KB |
| `README.md` | Documentation (this file) | ~15KB |
| `.gitignore` | Git ignore rules | ~1KB |
| `models/` | Trained model storage | Models not in repo |
| `datasets/` | Dataset information | READMEs only |

### What's Included

✅ **In Repository**:
- Complete Jupyter notebook with Gradio app
- Requirements and dependency specifications
- Comprehensive documentation
- Dataset download instructions
- Model storage structure

❌ **Not in Repository** (due to size):
- Trained model files (.h5) - ~80-100MB each
- Training datasets - Several GB
- Training history and logs

### Model File Management

Models are **gitignored** because:
- Large file size (~80-100MB each)
- GitHub has 100MB file size limit
- Better stored in Google Drive for Colab access

**To use models**:
1. Train models or obtain pre-trained .h5 files
2. Upload to Google Drive
3. Update `PROJECT_DIR` in notebook
4. Models load directly from Drive

---

## 🤝 Contributing

This is an educational FYP (Final Year Project). Contributions welcome:
- 🐛 Report bugs or issues
- 💡 Suggest new features or improvements
- 📚 Improve documentation
- ⭐ Star the repository if you find it helpful!

---

## 📄 License

This project is for educational purposes. Please respect:
- Dataset licenses (Kaggle terms of use)
- TensorFlow/Keras license (Apache 2.0)
- Gradio license (Apache 2.0)
- Use responsibly and ethically

---

## 🙏 Acknowledgments

- **Dataset Providers** - Kaggle community for making datasets publicly available
- **TensorFlow Team** - For the deep learning framework
- **Gradio Team** - For the amazing web interface library
- **Google Colab** - For providing free GPU resources
- **DenseNet Authors** - For the model architecture
- **Open-source Community** - For inspiration, tools, and support

---

## 👨‍💻 Project Information

**Project Name**: Chest X-ray Disease Detection System  
**Project Type**: Final Year Project (FYP)  
**Domain**: Medical Image Analysis + Deep Learning  
**Technology**: TensorFlow, DenseNet-121, Gradio  
**Deployment**: Google Colab + Public URL  
**Timeline**: October 2025 - January 2026  
**Status**: ✅ Complete and Ready for Demo!

---

## 🎯 Quick Links

- 📖 [Start Here - Deployment Guide](docs/START_HERE_DEPLOYMENT.md)
- ⚡ [5-Minute Quick Start](docs/QUICK_START_GRADIO.md)
- 📗 [Complete Deployment Guide](docs/DEPLOY_WITH_H5_MODELS.md)
- 🎨 [Visual Workflow](docs/VISUAL_WORKFLOW_GUIDE.md)
- 📔 [Main Notebook](notebooks/03_complete_colab_training_and_app.ipynb)

---

## 🌟 Features at a Glance

| Feature | Description |
|---------|-------------|
| 🤖 AI Model | DenseNet-121 with transfer learning |
| 🎨 Interface | Gradio web app with modern UI |
| ☁️ Platform | Google Colab (free GPU) |
| ⚡ Speed | 2-3 seconds per prediction |
| 🌐 Access | Public URL, shareable with anyone |
| 📱 Mobile | Fully responsive design |
| 🆓 Cost | Completely free to use! |
| 📊 Visualization | Confidence scores + bar charts |

---

## 📞 Support & Resources

### Internal Documentation
- All guides available in `docs/` folder
- Start with `START_HERE_DEPLOYMENT.md` for overview
- Use `QUICK_START_GRADIO.md` for quick reference
- Check `VISUAL_WORKFLOW_GUIDE.md` for diagrams

### External Resources
- [Gradio Documentation](https://www.gradio.app/docs/)
- [Google Colab User Guide](https://colab.research.google.com/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [DenseNet Paper](https://arxiv.org/abs/1608.06993)

### Getting Help
1. Check internal documentation first
2. Review troubleshooting sections
3. Try restarting Colab runtime
4. Re-run all cells from the beginning
5. Verify all file paths and names

---

## 🎉 Success Metrics

After following this guide, you should have:

✅ **5-minute deployment time**  
✅ **Working public Gradio URL**  
✅ **Accurate disease predictions**  
✅ **Professional web interface**  
✅ **Mobile-friendly responsive design**  
✅ **Easy sharing capability**  
✅ **Perfect demo for presentations!**

---

## 🏆 Project Achievements

- ✅ Implemented transfer learning with DenseNet-121
- ✅ Trained models for Pneumonia and COVID-19 detection
- ✅ Created professional Gradio web interface
- ✅ Deployed on Google Colab with public URL
- ✅ Complete documentation for deployment
- ✅ Mobile-friendly and accessible
- ✅ Ready for FYP demo and submission

---

**Last Updated**: October 30, 2025  
**Version**: 2.0 - Gradio Deployment Edition  
**Status**: 🚀 Ready for GitHub, Demo & Submission!

---

**⭐ If you find this project helpful, please star the repository!**  
**🔗 Share your deployed Gradio app URL with us!**  
**💬 Questions? Check the comprehensive docs folder!**

**Happy Deploying! 🎉**
