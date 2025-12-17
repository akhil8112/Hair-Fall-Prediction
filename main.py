import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# ==========================================
# 1. DATA LOADING & CLEANING
# ==========================================
filename = 'generated_1500_rows.xlsx'
try:
    df = pd.read_excel(filename)
    print("Data Loaded Successfully.")
except FileNotFoundError:
    print(f"Error: File '{filename}' not found.")
    exit()

# Rename columns for easier access
df.columns = [
    'Age', 'Gender', 'Family_History', 'Dandruff', 'Sleep_Hours', 'Stress_Level', 
    'Gut_Issues', 'Energy_Levels', 'Supplements', 'Hair_Fall_Count', 
    'Hair_Fall_Duration', 'Wash_Frequency', 'Iron_Levels'
]

# --- ROBUST TARGET MAPPING ---
# Strip whitespace to handle formatting issues (e.g. " 20 " vs "20")
df['Hair_Fall_Count'] = df['Hair_Fall_Count'].astype(str).str.strip()

# Map Target to Binary Risk (0 = Low Risk, 1 = High Risk)
risk_mapping = {
    '~ 20': 0, '20': 0, 
    '~ 40 - 50': 1, '~ 50 - 100': 1, '100+': 1
}
df['Risk_Target'] = df['Hair_Fall_Count'].map(risk_mapping)

# Drop rows that couldn't be mapped (clean data)
df = df.dropna(subset=['Risk_Target'])

# ==========================================
# 2. PREPROCESSING
# ==========================================
# Encode Categorical Features
le = LabelEncoder()
feature_cols = [
    'Gender', 'Family_History', 'Dandruff', 'Sleep_Hours', 'Stress_Level', 
    'Gut_Issues', 'Energy_Levels', 'Supplements', 'Hair_Fall_Duration', 
    'Wash_Frequency', 'Iron_Levels'
]

df_encoded = df.copy()
for col in feature_cols:
    df_encoded[col] = le.fit_transform(df[col].astype(str))

X = df_encoded.drop(['Hair_Fall_Count', 'Risk_Target'], axis=1)
y = df_encoded['Risk_Target']

# Scale Features (Critical for KNN, SVM, Clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================================
# OBJECTIVE 1: KEY FACTORS & EDA
# ==========================================
print("\n--- Objective 1: Identifying Key Factors ---")

# Feature Importance (Random Forest)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("Top 5 Influential Factors:")
print(feature_importance.head(5))

# Visualization 1: Correlation Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df_encoded.drop(['Hair_Fall_Count'], axis=1).corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# ==========================================
# OBJECTIVE 2: SUPERVISED ML MODELS
# ==========================================
print("\n--- Objective 2: Model Evaluation ---")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

best_model = None
best_acc = 0

print(f"{'Model':<25} | {'Accuracy':<10} | {'F1-Score':<10}")
print("-" * 50)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"{name:<25} | {acc*100:.2f}%     | {f1:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_model = model

# Visualization 2: Confusion Matrix for Best Model
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low Risk', 'High Risk'], yticklabels=['Low Risk', 'High Risk'])
plt.title(f'Confusion Matrix (Best Model)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ==========================================
# OBJECTIVE 3: CLUSTERING
# ==========================================
print("\n--- Objective 3: Clustering (User Segmentation) ---")

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# Visualization 3: PCA Cluster Plot
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(components[:, 0], components[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title('User Clusters (K-Means via PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(scatter, label='Cluster Group')
plt.show()

# ==========================================
# OBJECTIVE 4: EXPLAINABILITY (Global)
# ==========================================
print("\n--- Objective 4: Explainability ---")

# Visualization 4: Partial Dependence Plot (PDP) for Top Feature
top_feature = feature_importance.iloc[0]['Feature'] # e.g., Age
print(f"Generating Partial Dependence Plot for: {top_feature}")

fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(rf_model, X, [top_feature], ax=ax, kind='average')
ax.set_title(f'Feature Effect: {top_feature} vs Hair Fall Risk')
ax.grid(True, alpha=0.3)
plt.show()

# ==========================================
# OBJECTIVE 5: RISK SCORING & RECOMMENDATIONS
# ==========================================
print("\n--- Objective 5: Risk Scoring Framework ---")

# 1. Visualization 5: Risk Score Distribution
# Calculate risk scores for the test set
probs = best_model.predict_proba(X_test)[:, 1]
risk_scores = (probs * 100).astype(int)

plt.figure(figsize=(10, 6))
sns.histplot(x=risk_scores, hue=y_test.map({0: 'Low Risk', 1: 'High Risk'}), 
             kde=True, bins=20, palette='viridis', element="step")
plt.title('Distribution of Predicted Risk Scores')
plt.xlabel('Risk Score (0-100)')
plt.axvline(x=50, color='red', linestyle='--', label='Threshold')
plt.legend()
plt.show()

# 2. Recommendation Engine
def generate_patient_report(model, features_scaled, original_row):
    # Calculate Score
    prob = model.predict_proba([features_scaled])[0][1]
    score = int(prob * 100)
    
    # Generate Advice based on ORIGINAL data values
    advice = []
    
    # Stress Check
    if 'High' in str(original_row['Stress_Level']) or 'Moderate' in str(original_row['Stress_Level']):
        advice.append("Stress Management: Consider mindfulness, yoga, or therapy.")
        
    # Sleep Check
    if 'Less than 5 hours' in str(original_row['Sleep_Hours']):
        advice.append("Sleep Hygiene: Aim for 7-8 hours of quality sleep.")
        
    # Diet/Nutrition Check
    if 'Low' in str(original_row['Iron_Levels']):
        advice.append("Nutrition: Consult a doctor for Iron supplements.")
    
    # Scalp Health
    if 'Dandruff' in str(original_row['Dandruff']) and original_row['Dandruff'] != 'No':
        advice.append("Scalp Care: Use anti-dandruff treatments.")

    if not advice:
        advice.append("Lifestyle looks good! Maintain current habits.")
        
    return score, advice

# Example Report for a specific user
sample_idx = 0
sample_features = X_test[sample_idx]
original_idx = y_test.index[sample_idx] # Map back to original dataframe index
sample_original = df.loc[original_idx]

risk_score, recommendations = generate_patient_report(best_model, sample_features, sample_original)

print(f"\nSample Patient Report (ID: {original_idx})")
print(f"Predicted Risk Score: {risk_score}/100")
print("Recommendations:")
for item in recommendations:

    print(f"- {item}")
