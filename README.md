# Driver-drowsiness-MP_5
This project develops an AI-based system that detects driver fatigue using deep learning and image analysis.


Introduction

* Driver drowsiness is one of the major causes of road accidents. When drivers become sleepy, their reaction time and attention    decrease.
* This project develops an AI-based system that detects driver fatigue using deep learning and image analysis.
* The system analyzes driver facial images (eye and mouth states) and predicts whether the driver is alert or drowsy.

  Objectives:

* Detect driver fatigue using deep learning models
* Analyze eye closure and yawning patterns
* Build a CNN model and MobileNetV2 transfer learning model
* Classify driver state into multiple categories
* Improve road safety using AI.

Description:

Open -->	Eyes open (alert driver)
Closed -->	Eyes closed (possible drowsiness)
Yawn --> Driver yawning
No_yawn -->	Driver not yawning

Project Workflow:

Dataset Upload
        ↓
Dataset Extraction
        ↓
Dataset Exploration (EDA)
        ↓
Data Preprocessing
        ↓
Train-Test Split
        ↓
Data Augmentation
        ↓
CNN Model Training
        ↓
MobileNetV2 Transfer Learning
        ↓
Model Evaluation
        ↓
Driver Drowsiness Prediction

Dataset Extraction:
* Unzip the dataset
* Make images accessible for training.

Dataset Exploration:

* Image visualization
* Class distribution
* Random sample display

Dataset Splitting:

Training --> 70%
Validation --> 15%
Test --> 15%

CNN Model
* created a custom Convolutional Neural Network.
CNN learns patterns such as:
* Eye closure
* Mouth opening
* Facial fatigue patterns.

Transfer Learning (MobileNetV2):
* pretrained deep learning model.
1.Faster training
2.Higher accuracy
3.Uses features learned from large datasets

Final Prediction:

Prediction -->	Meaning
Open -->	Driver alert
Closed -->  Driver sleepy
Yawn -->	Driver yawning
No_yawn -->	Driver normal


Output Result:

* The model achieved 93% accuracy on unseen test data.
* This indicates the model can reliably detect driver fatigue.
* Transfer learning with MobileNetV2 improves performance compared to a basic CNN.


