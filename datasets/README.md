# Datasets Folder

This folder contains the datasets used for training the chest X-ray disease detection models.

## 📊 Required Datasets

### 1. Pneumonia Dataset
- **Source**: [Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Structure**: 
  ```
  pneumonia/
  └── chest_xray/
      ├── train/
      │   ├── NORMAL/
      │   └── PNEUMONIA/
      ├── val/
      │   ├── NORMAL/
      │   └── PNEUMONIA/
      └── test/
          ├── NORMAL/
          └── PNEUMONIA/
  ```

### 2. COVID-19 Dataset
- **Source**: [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- **Structure**: 
  ```
  covid_organized/
  ├── train/
  │   ├── Normal/
  │   └── COVID/
  ├── val/
  │   ├── Normal/
  │   └── COVID/
  └── test/
      ├── Normal/
      └── COVID/
  ```

## 📥 How to Download

1. **Create Kaggle Account**: Sign up at [Kaggle.com](https://www.kaggle.com/)

2. **Download Datasets**:
   - Pneumonia: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
   - COVID-19: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

3. **Extract Here**: Extract the downloaded datasets into this `datasets/` folder

## 📁 Expected Structure

After downloading and extracting, your folder should look like:

```
datasets/
├── pneumonia/
│   └── chest_xray/
│       ├── train/
│       ├── val/
│       └── test/
├── covid_organized/
│   ├── train/
│   ├── val/
│   └── test/
└── README.md (this file)
```

## 🎓 For Training

These datasets will be uploaded to Google Drive and accessed from the Colab notebook for training. See the main notebook for instructions on:
- Uploading datasets to Google Drive
- Organizing the data
- Setting correct paths in the notebook

## 📝 Note

- The datasets folder is typically NOT pushed to GitHub due to large size
- Store datasets locally or in Google Drive
- Reference them in the Colab notebook via Drive mount
- The `.gitignore` file should exclude these large files

## ⚠️ Important

- These are public datasets for educational use
- Respect dataset licenses and terms of use
- Include proper citations in your reports
- Use for research/educational purposes only
