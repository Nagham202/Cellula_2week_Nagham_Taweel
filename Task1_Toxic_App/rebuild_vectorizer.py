import re
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# the vectorizer wasn't saved in week 1, so this rebuilds it with the exact same steps


WEEK1_DIR = Path.home() / "Cellula_1week_Nagham_Taweel"
DATA_PATH = WEEK1_DIR / "toxic_data.csv"

VOCAB_PATH = Path(__file__).resolve().parent / "vocab.txt"
MODEL_PATH = Path(__file__).resolve().parent / "lstm_model.keras"


# same values as train_lstm.py
CLASSES_TO_DROP = ["Elections", "Sex-Related Crimes",
                   "Child Sexual Exploitation", "Suicide & Self-Harm"]
SEED = 42
MAX_LEN = 60


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


df = pd.read_csv(DATA_PATH)
df = df.rename(columns={"query": "text", "image descriptions": "image", "Toxic Category": "label"})
df = df[~df["label"].isin(CLASSES_TO_DROP)]
df["clean_text"] = df["text"].apply(clean_text) + " xxsep " + df["image"].apply(clean_text)
df = df.drop_duplicates(subset="clean_text")

class_names = sorted(df["label"].unique())
label_to_id = {name: i for i, name in enumerate(class_names)}
df["label_id"] = df["label"].map(label_to_id)

# .values breaks train_test_split here because of pyarrow, this gives a normal array
X = df["clean_text"].to_numpy(dtype=object)
y = df["label_id"].to_numpy()

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED)
print("Training rows:", len(X_train))

# train only, like week 1
vectorizer = tf.keras.layers.TextVectorization(
    standardize=None, split="whitespace", output_mode="int", output_sequence_length=MAX_LEN)
vectorizer.adapt(X_train)

vocab = vectorizer.get_vocabulary()
print("Vocabulary size:", len(vocab))

# skip "" and [UNK], keras adds them back when loading
with open(VOCAB_PATH, "w") as f:
    for word in vocab[2:]:
        f.write(word + "\n")
print("Saved", VOCAB_PATH)


# check: the saved vocab + model should give the same validation f1 as week 1
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

with open(VOCAB_PATH) as f:
    saved_words = f.read().splitlines()

loaded_vectorizer = tf.keras.layers.TextVectorization(
    standardize=None, split="whitespace", output_mode="int",
    output_sequence_length=MAX_LEN, vocabulary=saved_words)

model = tf.keras.models.load_model(MODEL_PATH)

X_val_seq = loaded_vectorizer(X_val).numpy()
val_predictions = model.predict(X_val_seq, verbose=0).argmax(axis=1)

print("Validation macro F1:", round(f1_score(y_val, val_predictions, average="macro"), 4))
print("Week 1 value was:0.8648")
