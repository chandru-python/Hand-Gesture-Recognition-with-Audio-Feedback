import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

# =====================================
# Load dataset
# =====================================

df = pd.read_csv("hand_landmarks.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# =====================================
# Encode labels
# =====================================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

joblib.dump(encoder, "label_encoder.pkl")

y_categorical = to_categorical(y_encoded)

# =====================================
# Split data
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

# =====================================
# Build model
# =====================================

model = Sequential()

model.add(Dense(256, activation="relu", input_shape=(63,)))
model.add(Dropout(0.3))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(y_categorical.shape[1], activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =====================================
# Train model
# =====================================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32
)

# =====================================
# Save model
# =====================================

model.save("gesture_model.h5")

# =====================================
# Plot Accuracy Graph
# =====================================

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"])
plt.show()

# =====================================
# Plot Loss Graph
# =====================================

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"])
plt.show()

# =====================================
# Predictions
# =====================================

y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# =====================================
# Confusion Matrix
# =====================================

cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =====================================
# Classification Report
# =====================================

report = classification_report(
    y_true_classes,
    y_pred_classes,
    target_names=encoder.classes_,
    output_dict=True
)

print("\nClassification Report:\n")
print(classification_report(
    y_true_classes,
    y_pred_classes,
    target_names=encoder.classes_
))

# =====================================
# Extract Precision, Recall, F1
# =====================================

precision = []
recall = []
f1 = []
labels = []

for key in encoder.classes_:

    precision.append(report[key]["precision"])
    recall.append(report[key]["recall"])
    f1.append(report[key]["f1-score"])
    labels.append(key)

# =====================================
# Precision Graph
# =====================================

plt.figure()
plt.bar(labels, precision)
plt.title("Precision per Class")
plt.ylabel("Precision")
plt.xlabel("Class")
plt.show()

# =====================================
# Recall Graph
# =====================================

plt.figure()
plt.bar(labels, recall)
plt.title("Recall per Class")
plt.ylabel("Recall")
plt.xlabel("Class")
plt.show()

# =====================================
# F1 Score Graph
# =====================================

plt.figure()
plt.bar(labels, f1)
plt.title("F1 Score per Class")
plt.ylabel("F1 Score")
plt.xlabel("Class")
plt.show()

# =====================================
# Overall metrics
# =====================================

print("\nOverall Accuracy:",
      np.mean(y_pred_classes == y_true_classes))

print("\nTraining Complete")