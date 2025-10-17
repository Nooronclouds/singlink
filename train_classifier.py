import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU

import pickle
import numpy as np

# Try different import approach to avoid version conflicts
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report
except ImportError as e:
    print(f"Import error: {e}")
    print("Try: pip install numpy==1.24.3 scikit-learn==1.3.0")
    exit()

# Load your data
print("Loading data...")
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

print(f"Dataset shape: {data.shape}")
print(f"Number of samples: {len(data)}")
print(f"Number of classes: {len(set(labels))}")

# Check class distribution
unique, counts = np.unique(labels, return_counts=True)
print("Samples per class:")
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count} samples")

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, 
    test_size=0.2, 
    shuffle=True, 
    stratify=labels,
    random_state=42
)

print(f"Training set: {x_train.shape}")
print(f"Test set: {x_test.shape}")

# Use optimized Random Forest
model = RandomForestClassifier(
    n_estimators=150,       # More trees for better accuracy
    max_depth=15,           # Prevent overfitting
    min_samples_split=5,    # Better generalization
    min_samples_leaf=2,     # Better generalization  
    random_state=42,
    n_jobs=-1               # Use all CPU cores
)

print("Training improved Random Forest...")
model.fit(x_train, y_train)

# Predict and evaluate
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 **IMPROVED ACCURACY: {accuracy * 100:.2f}%**")

# Show per-class performance
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation for more reliable score
print("\nRunning cross-validation...")
cv_scores = cross_val_score(model, data, labels, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean() * 100:.2f}%")

# Save model
print("\nSaving model...")
with open('model_enhanced.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("✅ Enhanced model saved as 'model_enhanced.p'")