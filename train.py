# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load dataset
df = pd.read_csv("StressLevelDataset.csv")

# Define features and target
feature_cols = [
    "anxiety_level", "self_esteem", "mental_health_history", "depression",
    "headache", "blood_pressure", "sleep_quality", "breathing_problem",
    "noise_level", "living_conditions", "safety", "basic_needs",
    "academic_performance", "study_load", "teacher_student_relationship",
    "future_career_concerns", "social_support", "peer_pressure",
    "extracurricular_activities", "bullying"
]
target_col = "stress_level"

X = df[feature_cols]
y = df[target_col]

# Encode target labels if categorical
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nModel Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_.astype(str)))

# Save with joblib (compressed, version-friendly)
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/stress_model.joblib", compress=3)
joblib.dump(label_encoder, "models/label_encoder.joblib", compress=3)

print("\n Model and encoder saved in 'models/' folder.")

# (Optional) Export to ONNX for maximum compatibility
try:
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [(name, FloatTensorType([None, 1])) for name in feature_cols]
    onnx_model = convert_sklearn(model, initial_types=[("input", FloatTensorType([None, len(feature_cols)]))])
    with open("models/stress_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("ONNX model exported successfully.")
except ImportError:
    print(" skl2onnx not installed, skipping ONNX export.")
