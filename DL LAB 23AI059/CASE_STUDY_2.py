import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

normal_data = np.random.normal(loc=0.0, scale=1.0, size=(1000, 20))
fraud_data = np.random.normal(loc=5.0, scale=1.0, size=(50, 20))

X = np.vstack([normal_data, fraud_data])
y = np.hstack([np.zeros(1000), np.ones(50)])


X_train = normal_data
X_test = X
y_test = y

input_dim = X_train.shape[1]

autoencoder = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(input_dim, activation="linear")
])

autoencoder.compile(optimizer="adam", loss="mse")

history = autoencoder.fit(
    X_train, X_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    verbose=2
)

reconstructions = autoencoder.predict(X_test, verbose=0)
mse = np.mean(np.square(X_test - reconstructions), axis=1)

train_recon = autoencoder.predict(X_train, verbose=0)
train_mse = np.mean(np.square(X_train - train_recon), axis=1)
threshold = np.mean(train_mse) + 2*np.std(train_mse)

y_pred = (mse > threshold).astype(int)

from sklearn.metrics import classification_report, confusion_matrix

print("Threshold:", threshold)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

plt.hist(mse[y_test==0], bins=30, alpha=0.7, label="Normal")
plt.hist(mse[y_test==1], bins=30, alpha=0.7, label="Fraud")
plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
plt.legend()
plt.xlabel("Reconstruction Error")
plt.ylabel("Count")
plt.title("Autoencoder Outlier Detection")
plt.show()
