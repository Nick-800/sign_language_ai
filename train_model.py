import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv("gesture_dataset.csv")

X = data.iloc[:, :-1].values   # features (63 values)
y = data.iloc[:, -1].values    # labels

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
np.save("gesture_labels.npy", encoder.classes_)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=60, batch_size=32)

# Save
model.save("gesture_model_tf.h5")
print("✅ Model trained and saved as gesture_model_tf.h5")
