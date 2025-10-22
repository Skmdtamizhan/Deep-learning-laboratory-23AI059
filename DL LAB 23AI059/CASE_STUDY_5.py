import numpy as np
from sklearn.neural_network import MLPClassifier


X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

model = MLPClassifier(hidden_layer_sizes=(2,), activation='tanh', solver='adam', max_iter=10000, random_state=42)

model.fit(X, y)

y_pred = model.predict(X)

print("Input:\n", X)
print("Predicted Output:", y_pred)
print("Expected Output :", y)

print("Training Accuracy:", model.score(X, y))
