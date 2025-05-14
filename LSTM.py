import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Load dataset
file_path =(r"C:\Users\saksh\Desktop\Minor Project\combined_emotion.csv")
df = pd.read_csv(file_path)

# Balance dataset
majority_classes = ["joy", "sad"]
minority_classes = ["anger", "fear", "love", "suprise"]

df_majority = df[df["emotion"].isin(majority_classes)]
df_minority = [df[df["emotion"] == emo] for emo in minority_classes]
df_minority_upsampled = [resample(d, replace=True, n_samples=121187, random_state=42) for d in df_minority]
df_balanced = pd.concat([df_majority] + df_minority_upsampled).sample(frac=1, random_state=42).reset_index(drop=True)

# Preprocessing
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 512

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(df_balanced["sentence"].astype(str))
sequences = tokenizer.texts_to_sequences(df_balanced["sentence"].astype(str))
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df_balanced["emotion"])
num_classes = len(np.unique(labels))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Build Optimized LSTM model
model = Sequential([
    Embedding(MAX_VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=128, verbose=1)

# Save model and tokenizer
model.save("lstm_sentiment_model.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Model training complete and saved with optimized parameters.")

# Evaluate model
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Overall Test Accuracy: {accuracy * 100:.2f}%")
print(f"Overall Test Precision: {precision * 100:.2f}%")

# Classification report for individual emotions
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
print("\nClassification Report:\n", class_report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()