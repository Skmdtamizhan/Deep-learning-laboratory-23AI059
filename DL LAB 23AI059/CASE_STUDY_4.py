import numpy as np

ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [0, 0, 5, 4],
    [0, 1, 5, 4]
], dtype=np.float32)

num_users, num_movies = ratings.shape
num_hidden = 2
learning_rate = 0.1
epochs = 5000

np.random.seed(42)
W = np.random.normal(0, 0.01, size=(num_movies, num_hidden))
hb = np.zeros(num_hidden)
vb = np.zeros(num_movies)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

for epoch in range(epochs):
    for u in range(num_users):
        v0 = ratings[u]
        mask = (v0 > 0)

        h_prob = sigmoid(np.dot(v0, W) + hb)
        h_state = (h_prob > np.random.rand(num_hidden)).astype(float)

        v_prob = sigmoid(np.dot(h_state, W.T) + vb)
        v_prob = v_prob * mask
        h_prob_neg = sigmoid(np.dot(v_prob, W) + hb)
        W += learning_rate * (np.outer(v0, h_prob) - np.outer(v_prob, h_prob_neg))
        vb += learning_rate * (v0 - v_prob)
        hb += learning_rate * (h_prob - h_prob_neg)

predicted_ratings = sigmoid(np.dot(sigmoid(np.dot(ratings, W) + hb), W.T) + vb)

print("Original Ratings (0=unrated):\n", ratings)
print("\nPredicted Ratings:\n", np.round(predicted_ratings,2))

print("\nTop Recommended Movie for Each User:")
for user_id in range(num_users):
    user_unrated = np.where(ratings[user_id] == 0)[0]
    if len(user_unrated) == 0:
        print(f"User {user_id}: All movies rated")
        continue
    predicted_for_unrated = predicted_ratings[user_id, user_unrated]
    recommended_idx = user_unrated[np.argmax(predicted_for_unrated)]
    recommended_score = predicted_for_unrated[np.argmax(predicted_for_unrated)]
    print(f"User {user_id}: Recommend Movie {recommended_idx} (Predicted Rating {recommended_score:.2f})")
