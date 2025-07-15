import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_csv("exams.csv")

# Encode categorical columns
df['gender'] = df['gender'].map({'female': 0, 'male': 1})
df['race/ethnicity'] = df['race/ethnicity'].map({
    'group A': 0, 'group B': 1, 'group C': 2, 'group D': 3, 'group E': 4
})
df['parental level of education'] = df['parental level of education'].map({
    "associate's degree": 0, "bachelor's degree": 1, "high school": 2,
    "master's degree": 3, "some college": 4, "some high school": 5
})
df['lunch'] = df['lunch'].map({'free/reduced': 0, 'standard': 1})
df['test preparation course'] = df['test preparation course'].map({'completed': 0, 'none': 1})

# Split data
X = df[['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course', 'reading score', 'writing score']]
y = df['math score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = ExtraTreesRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "math_score_predictor.pkl")
print("✅ Model retrained and saved successfully!")
