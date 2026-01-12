# Waste Classifier Model Training

This directory contains everything needed to train a custom waste classification model.

## 📁 Directory Structure

```
ml_training/
├── dataset/                    # Training images
│   ├── organic/               # Food waste, leaves, compost
│   ├── recyclable/            # Plastic, paper, glass, metal
│   ├── hazardous/             # Batteries, chemicals, paint
│   ├── ewaste/                # Electronics, cables, phones
│   └── general/               # Mixed/general waste
├── train_model.py             # Training script
├── convert_to_tflite.py       # Convert to TFLite for mobile
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Set Up Environment
```bash
cd ml_training
python -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### 2. Prepare Dataset
Add images to each category folder:
- **organic/**: ~500+ images of food scraps, leaves, plants
- **recyclable/**: ~500+ images of bottles, cans, paper, cardboard
- **hazardous/**: ~500+ images of batteries, chemicals, paint cans
- **ewaste/**: ~500+ images of phones, cables, electronics
- **general/**: ~500+ images of mixed waste

**Tips for good dataset:**
- Use varied lighting conditions
- Include different angles
- Mix indoor/outdoor photos
- Include items in bins and standalone
- Aim for at least 500 images per category

### 3. Train the Model
```bash
python train_model.py
```

### 4. Convert to TFLite (for mobile)
```bash
python convert_to_tflite.py
```

### 5. Copy Model to Flutter App
```bash
cp models/waste_classifier.tflite ../smart_waste_app/assets/models/
```

## 📊 Dataset Sources

You can download waste datasets from:
1. **TrashNet**: https://github.com/garythung/trashnet
2. **Waste Classification Data**: https://www.kaggle.com/datasets/techsash/waste-classification-data
3. **TACO Dataset**: http://tacodataset.org/

## 🎯 Model Performance

After training, you should see:
- Training accuracy: >90%
- Validation accuracy: >85%
- Test accuracy: >80%

## 📱 Integration with Flutter App

The trained model will be integrated with:
- `lib/services/ml/tflite_classifier_service.dart`
- Model file: `assets/models/waste_classifier.tflite`
- Labels file: `assets/models/waste_labels.txt`
