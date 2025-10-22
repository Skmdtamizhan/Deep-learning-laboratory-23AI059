import librosa
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from google.colab import files
import requests
import soundfile as sf

# 3️⃣ Function: Download a Proper Sample Audio (.wav)
def download_sample_audio():
    url = "https://github.com/Jakobovski/free-spoken-digit-dataset/raw/master/recordings/0_jackson_0.wav"
    r = requests.get(url)
    with open("sample.wav", "wb") as f:
        f.write(r.content)
    print("✅ Sample audio file downloaded successfully: sample.wav")
    return "sample.wav"

# 4️⃣ Audio Preprocessing
def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)  # Resample to 16kHz
    return y, sr

# 5️⃣ Load Pretrained Model (English)
model_name = "facebook/wav2vec2-base-960h"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
model.eval()

# 6️⃣ Transcription Function
def transcribe_audio(file_path):
    y, sr = preprocess_audio(file_path)
    inputs = tokenizer(y, return_tensors="pt", padding="longest")

    with torch.no_grad():
        logits = model(input_values=inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)
    return transcription[0]

# 7️⃣ Choose File (Sample or Upload)
choice = input("Type 'sample' to use test audio or 'upload' to use your own file: ").strip().lower()

if choice == "sample":
    file_path = download_sample_audio()
else:
    print("📁 Please upload your .wav file")
    uploaded = files.upload()
    file_path = list(uploaded.keys())[0]

# 8️⃣ Run Transcription
print("\n🔍 Processing audio...")
try:
    transcription = transcribe_audio(file_path)
    print("\n🗣️ Transcription:\n", transcription)
except Exception as e:
    print("\n❌ Error while processing audio:", e)
