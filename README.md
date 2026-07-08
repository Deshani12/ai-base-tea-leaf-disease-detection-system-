# 🌱 AI-Based Tea Leaf Disease Prediction and Control System for Sri Lankan Farmers 

## 📌 Project Overview

Tea cultivation remains a vital component of Sri Lanka’s agricultural economy, contributing significantly to export earnings and rural employment. However, tea plantation productivity and crop quality are frequently affected by various diseases and pest infestations, including **Brown Blight, Gray Blight, Algal Leaf Spot, Helopeltis, Red Spider, and Green Mirid Bug**.

Traditional disease identification methods mainly depend on manual observation and expert consultation. These approaches are time-consuming, subjective, and often difficult for small-scale farmers due to limited access to agricultural experts.

This project introduces an **AI-Based Tea Leaf Disease Prediction and Control System** that assists Sri Lankan tea farmers by providing automated disease identification and treatment recommendations using Artificial Intelligence and Deep Learning technologies.

The system uses a **Convolutional Neural Network (CNN)** model trained with labeled tea leaf images to classify seven different tea leaf conditions. Farmers can upload a tea leaf image through the web application, receive an instant disease prediction with confidence level, and obtain recommended control methods.

---

# 🎯 Research Objectives

The main objectives of this project are:

- Develop an AI-based system for automatic tea leaf disease detection.
- Reduce dependency on manual disease identification methods.
- Provide early disease prediction to minimize crop losses.
- Recommend suitable disease control practices.
- Maintain prediction history for future monitoring and analysis.
- Support smart agriculture practices in Sri Lanka.

---

# ✨ Key Features

## 🌿 AI Tea Leaf Disease Detection

The system identifies seven tea leaf conditions using a trained CNN deep learning model.

Supported categories:

| Category | Type |
|---|---|
| Healthy Leaf | Normal condition |
| Brown Blight | Disease |
| Gray Blight | Disease |
| Tea Algal Leaf Spot | Disease |
| Helopeltis | Pest |
| Red Spider | Pest |
| Green Mirid Bug | Pest |

---

## 📷 Image-Based Prediction

Users can:

- Upload tea leaf images.
- Automatically preprocess images.
- Generate disease predictions.
- View prediction confidence percentage.

---

## 🧠 Deep Learning Model

The prediction model was developed using:

- TensorFlow
- Keras
- Convolutional Neural Network (CNN)

### Model Performance

- Dataset: Labeled tea leaf image dataset
- Image Classification: 7 classes
- Achieved Accuracy: **89.4%**

The model applies:

- Image resizing
- Normalization
- Data augmentation
- CNN feature extraction
- Classification

---

# 🏗 System Architecture

             Farmer/User
                 |
                 |
         Upload Tea Leaf Image
                 |
                 |
         Flask Web Application
                 |
                 |
      Image Preprocessing Module
                 |
                 |
          CNN Deep Learning Model
                 |
                 |
        Disease Prediction Result
                 
---

# 🛠 Technologies Used

## Frontend Development

- HTML5
- Tailwind CSS
- JavaScript
- Chart.js

## Backend Development

- Python
- Flask Framework

## Artificial Intelligence

- TensorFlow
- Keras
- CNN Architecture
- Computer Vision

## Database

- SQLite

## Development Tools

- Visual Studio Code
- Google Colab
- Git & GitHub

---

# 📂 Project Structure

  ```
TEA_LEAF_DISEASE_DETECTION_SYSTEM/
│
├── dataset/                          # Training dataset
│   ├── Brown_Blight/
│   ├── Gray_Blight/
│   ├── Green_Mirid_Bug/
│   ├── Healthy_Leaf/
│   ├── Helopeltis/
│   ├── Red_Spider/
│   └── Tea_algal_leaf_spot/
│
├── static/                           # Static assets
│   ├── image/                        # Sample tea leaf images
│   └── uploads/                      # User-uploaded images
│
├── templates/                        # HTML templates
│   ├── index.html                    # Main prediction page
│   └── history.html                  # Prediction history page
│
├── model/                            # Trained deep learning model
│   └── tea_leaf_disease_model.h5
│
├── app.py                            # Main Flask application
├── train_model.py                    # Model training script
├── migrate_db.py                     # Database initialization/migration
│
├── predictions.db                    # SQLite database for prediction history
├── class_labels.txt                  # Disease class labels
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
│
└── .gitignore                        
```



---
# 📊 Database Integration

The system uses SQLite database integration to store:

Uploaded image information
Predicted disease category
Prediction confidence
Prediction date and time

This allows users to review previous predictions and analyze disease patterns.

# 🌱 Disease Control Recommendations

After prediction, the system provides recommendations such as:

Disease prevention methods.
Suitable management practices.
Early intervention guidance.
Reduction of unnecessary pesticide usage.
🔬 Research Contribution

This project contributes to smart agriculture by combining:

Artificial Intelligence
Deep Learning
Web Development
Database Management

The proposed solution provides a scalable and cost-effective approach for tea disease monitoring and supports Sri Lanka’s transition toward digital agriculture.

---

# 👩‍💻 Project Information
Project Title

AI-Based Tea Leaf Disease Prediction and Control System for Sri Lankan Farmers

Research Area
Artificial Intelligence
Deep Learning
Computer Vision
Smart Agriculture
Model Accuracy

89.4%

---

# 🖥 System Screenshots

  Home Page
<img width="602" height="292" alt="image" src="https://github.com/user-attachments/assets/1c918069-0cb3-4908-adb6-e2a4d2901399" />

  Disease Prediction Result
<img width="606" height="301" alt="image" src="https://github.com/user-attachments/assets/fb63555d-4f30-42d9-92ec-377c1ee64fa3" />

  Prediction History Dashboard
<img width="606" height="287" alt="image" src="https://github.com/user-attachments/assets/381ab377-7108-4a44-a375-aa357ac50c11" />

---


# ⚙️ Installation and Setup

## 1. Clone Repository

```bash
git clone https://github.com/Deshani12/ai-base-tea-leaf-disease-detection-system-.git

#Navigate into project:
cd tea-leaf-disease-detection-system

#Create Virtual Environment
python -m venv venv                                   

#Activate environment:
venv\Scripts\activate

#Install Required Libraries
pip install -r requirements.txt

#Run Flask Application
python app.py

#Open Browser
http://127.0.0.1:5000




