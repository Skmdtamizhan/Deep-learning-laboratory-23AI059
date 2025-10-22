
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# 3️⃣ Data Preparation
text = "Your dataset text goes here. It can be a large corpus of text."

# Tokenize at character level
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts([text])

# Convert text to sequence of integers
sequences = tokenizer.texts_to_sequences([text])[0]

# Create input-output pairs
sequence_length = 40
X, y = [], []

for i in range(sequence_length, len(sequences)):
    X.append(sequences[i - sequence_length:i])
    y.append(sequences[i])

X = np.array(X)
y = np.array(y)

# Pad input sequences
X = pad_sequences(X, maxlen=sequence_length)

# 4️⃣ Build LSTM Model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 50

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=sequence_length),
    LSTM(128, return_sequences=True),
    LSTM(128),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 5️⃣ Train the Model
y = np.expand_dims(y, axis=-1)
model.fit(X, y, epochs=10, batch_size=64)

# 6️⃣ Text Generation Function
def generate_text(model, tokenizer, seed_text, num_chars=100):
    for _ in range(num_chars):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=sequence_length, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)[0]
        output_char = ""
        for char, index in tokenizer.word_index.items():
            if index == predicted:
                output_char = char
                break
        seed_text += output_char
    return seed_text

# 7️⃣ Generate New Text
print("\n🧾 Generated Text:\n")
print(generate_text(model, tokenizer, seed_text="Your dataset text"))