
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convert labels to categorical (one-hot encoding)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy :", test_acc)

# Sample predictions
import numpy as np
predictions = model.predict(x_test[:5])
print("\nSample Predictions (Cheque Digits):")
print("Predicted :", np.argmax(predictions, axis=1))
print("Actual    :", np.argmax(y_test[:5], axis=1))

# Result interpretation
print("\nResult : The multi-layer neural network correctly learns to recognize handwritten digits.")
