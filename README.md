# Hair Fall Prediction using Machine Learning

## Overview
The Hair Fall Prediction System is a Machine Learning project that analyzes lifestyle, health, and genetic factors to predict the risk of hair fall. The system classifies individuals into Low Risk or High Risk categories, identifies important factors affecting hair fall, segments users into clusters, and generates personalized recommendations.

This project demonstrates a complete end-to-end Machine Learning pipeline including data preprocessing, model training, evaluation, clustering, explainability, and recommendation system.

---

## Objectives
- Identify key factors causing hair fall
- Train and compare multiple Machine Learning models
- Predict hair fall risk (Low / High)
- Generate risk score (0–100)
- Segment users using clustering
- Provide personalized recommendations

---

## Features
- Data Cleaning and Preprocessing
- Feature Encoding and Scaling
- Feature Importance Analysis
- Multiple ML Model Training
- Model Evaluation (Accuracy & F1 Score)
- Best Model Selection
- Confusion Matrix Visualization
- K-Means Clustering for User Segmentation
- PCA for Visualization
- Partial Dependence Plot for Explainability
- Risk Score Prediction
- Personalized Recommendation System

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Machine Learning Models Used
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting
- K-Means Clustering
- PCA (Principal Component Analysis)

---

## Dataset Features
The dataset includes the following features:
- Age
- Gender
- Family History
- Dandruff
- Sleep Hours
- Stress Level
- Gut Issues
- Energy Levels
- Supplements
- Hair Fall Duration
- Wash Frequency
- Iron Levels

### Target Variable
Hair fall risk is categorized as:
- 0 → Low Risk
- 1 → High Risk

---

## Project Workflow
1. Data Loading
2. Data Cleaning
3. Target Mapping
4. Encoding Categorical Features
5. Feature Scaling
6. Model Training
7. Model Evaluation
8. Best Model Selection
9. Clustering using K-Means
10. PCA Visualization
11. Model Explainability
12. Risk Score Generation
13. Recommendation Generation

---

## Installation and Setup

### Step 1: Clone Repository
git clone https://github.com/your-username/hair-fall-prediction.git  
cd hair-fall-prediction

### Step 2: Install Dependencies
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl

### Step 3: Add Dataset
Place the dataset file in the project folder:
generated_1500_rows.xlsx

### Step 4: Run the Project
python main.py

---

## Project Structure
hair-fall-prediction/
│
├── generated_1500_rows.xlsx
├── main.py
├── README.md

---

## Output
The system provides:
- Hair Fall Risk (Low / High)
- Risk Score (0–100)
- Important Features Affecting Hair Fall
- User Cluster Group
- Personalized Recommendations

---

## Risk Score Interpretation
| Risk Score | Category |
|------------|----------|
| 0 – 49     | Low Risk |
| 50 – 100   | High Risk |

---

## Example Recommendations
- Reduce stress through meditation or yoga
- Improve sleep quality (7–8 hours)
- Increase iron intake
- Use anti-dandruff treatment if needed

---

## Future Improvements
- Deploy as Web Application
- Add Deep Learning Models
- Use Real Medical Dataset
- Improve Recommendation Engine
- Build Mobile App Version

---

## Author
Akhil Singh

---

## If you like this project
Give this repository a star on GitHub ⭐
