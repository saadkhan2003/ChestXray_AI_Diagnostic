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

## 📊 Datasets Used

### 1. Pneumonia Detection
- **Source**: [Chest X-ray Images (Pneumonia) on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes**: Normal, Pneumonia
- **Size**: ~5,863 images

### 2. COVID-19 Detection
- **Source**: [COVID-19 Radiography Database on Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- **Classes**: Normal, COVID-19
- **Size**: ~21,165 images

---

## 💻 Technical Stack

### Core Technologies
- **TensorFlow/Keras** - Deep learning framework
- **DenseNet-121** - Pre-trained CNN model
- **Gradio** - Web interface library
- **Google Colab** - Free cloud GPU platform

### Key Libraries
```
tensorflow>=2.8.0
gradio>=3.0.0
numpy
pandas
matplotlib
opencv-python-headless
pillow
scikit-learn
seaborn
```

---

## 🎓 For FYP/Academic Projects

### What to Include in Report
1. ✅ **System Architecture** - DenseNet-121, Gradio, Colab
2. ✅ **Deployment Process** - Screenshots and steps
3. ✅ **Results** - Predictions with confidence scores
4. ✅ **Accessibility** - Public URL demonstration
5. ✅ **Testing** - Multiple X-ray examples

### Demo Preparation Checklist
- [ ] Practice 5-minute demo
- [ ] Prepare 3-5 test X-ray images
- [ ] Have public URL ready and tested
- [ ] Take backup screenshots
- [ ] Include medical disclaimer
- [ ] Prepare answers for technical questions

### Presentation Flow
1. Introduce the problem (chest X-ray diseases)
2. Explain the solution (AI detection system)
3. Show the Colab notebook structure
4. Explain transfer learning approach
5. Demo the live Gradio app
6. Upload different X-ray types
7. Explain confidence scores and predictions
8. Discuss accessibility via public URL

---

## ⚠️ Medical Disclaimer

**IMPORTANT**: This project is for **educational and research purposes ONLY**. 

- ❌ NOT intended for clinical use
- ❌ NOT a medical diagnostic tool
- ❌ NOT a replacement for professional medical advice
- ✅ Educational demonstration of ML concepts
- ✅ Research and learning purposes

**Always consult qualified healthcare professionals for medical diagnosis.**

---

## 🐛 Troubleshooting

### Common Issues & Solutions

**Q: Models not loading?**
- Check file names match exactly: `pneumonia_model_best.h5` and `covid_model_best.h5`
- Ensure files are uploaded to `/content/models/` folder
- Try re-uploading the files
- Verify file sizes (should be ~80-100MB each)

**Q: Public URL not working?**
- Keep Colab tab/window open
- Don't stop the launch cell execution
- Re-run the launch cell if link expires
- Check internet connection

**Q: Predictions are slow?**
- Enable GPU in Colab (Runtime → Change runtime type → GPU)
- Restart runtime if needed
- Ensure images are reasonable size (<5MB)

**Q: "Out of memory" error?**
- Runtime → Restart runtime
- Re-run all cells from beginning
- Upload one image at a time

**For detailed help:** Check the troubleshooting sections in documentation files in `docs/` folder

---

## 📦 Repository Contents

### Essential Files
- ✅ `notebooks/03_complete_colab_training_and_app.ipynb` - Main Colab notebook
- ✅ `requirements.txt` - Python dependencies for Colab
- ✅ `README.md` - This file
- ✅ `.gitignore` - Git ignore rules

### Documentation (`docs/` folder)
- ✅ Complete deployment guides
- ✅ Quick start references
- ✅ Visual workflow diagrams
- ✅ Troubleshooting tips and FAQs

### Models Folder
- Place your trained `.h5` model files here (not pushed to GitHub due to size)
- `pneumonia_model_best.h5`
- `covid_model_best.h5`

---

## 🚀 Deployment Options

### Option 1: Quick Deploy (Recommended)
**Use pre-trained models**
- Upload your `.h5` models to Colab
- Run the notebook
- Get public URL in 5 minutes

### Option 2: Full Pipeline
**Train and deploy**
- Use the training sections in notebook
- Train on Colab GPU (~5-6 hours)
- Automatically deploy after training
- Get public URL

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
